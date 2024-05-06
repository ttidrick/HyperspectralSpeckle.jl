using FourierTools
using Distributions


@views function calculate_layer_masks_iso!(atmosphere, observations, object, masks)
    FTYPE = gettype(atmosphere)
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    scaleby_height = 1 .- atmosphere.heights ./ object.height
    atmosphere_masks = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ, Threads.nthreads())
    Threads.@threads :static for t=1:observations.nepochs
        tid = Threads.threadid()
        for w=1:atmosphere.nλ
            for l=1:atmosphere.nlayers
                center = Tuple(atmosphere.positions[:, t, l, w])
                deextractor = create_extractor_adjoint(center, atmosphere.dim, observations.dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                atmosphere_masks[:, :, l, w, tid] .+= deextractor*masks.masks[:, :, 1, w]
            end
        end
    end
    atmosphere.masks = dropdims(sum(atmosphere_masks, dims=5), dims=5)
    atmosphere.masks[atmosphere.masks .> 0] .= 1
end

@views function calculate_composite_pupil_isoplanatic!(atmosphere, observations, object, masks; propagate=false)
    FTYPE = gettype(atmosphere)

    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    scaleby_height = 1 .- atmosphere.heights ./ object.height

    atmosphere.A = ones(FTYPE, observations.dim, observations.dim, observations.nepochs, atmosphere.nλ)
    atmosphere.ϕ_composite = zeros(FTYPE, observations.dim, observations.dim, observations.nepochs, atmosphere.nλ)
    atmosphere.ϕ_slices = zeros(FTYPE, observations.dim, observations.dim, atmosphere.nlayers, observations.nepochs, atmosphere.nλ)
    atmosphere.mask = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)

    Threads.@threads for t=1:observations.nepochs
        for w=1:atmosphere.nλ
            for l=1:atmosphere.nlayers
                extractor = create_extractor_operator(atmosphere.positions[:, t, l, w], atmosphere.dim, observations.dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                position2phase!(atmosphere.ϕ_slices[:, :, l, t, w], atmosphere.ϕ[:, :, l, w], extractor)
                atmosphere.ϕ_composite[:, :, t, w] .+= masks.masks[:, :, 1, w] .* atmosphere.ϕ_slices[:, :, l, t, w]
            end
        end
    end
end

@views function calculate_smoothed_opd(atmosphere, observations)
    FTYPE = gettype(atmosphere)

    ϕ_ref = (FTYPE(2pi)/atmosphere.λ_ref) .* atmosphere.opd
    ϕ_smooth = zeros(FTYPE, size(ϕ_ref))
    for l=1:atmosphere.nlayers
        sampling_ref = atmosphere.sampling_nyquist[l] * (atmosphere.λ_ref / atmosphere.λ_nyquist)
        D_ref_pix = round(Int64, observations.D / sampling_ref)
        mask = make_simple_mask(atmosphere.dim, D_ref_pix)
        smooth_to_rmse!(ϕ_smooth[:, :, l], ϕ_ref[:, :, l], 0.2, mask, size(ϕ_ref, 1), FTYPE=FTYPE)
        # filter_to_rmse!(ϕ_smooth[:, :, l], ϕ_ref[:, :, l], 0.2, mask, size(ϕ_ref, 1), FTYPE=FTYPE)
    end

    opd_smooth = ϕ_smooth .* (atmosphere.λ_ref/FTYPE(2pi))
    return opd_smooth
end

@views function create_images_isoplanatic!(observations, atmosphere, masks, object; build_dim=256, noise=false)
    FTYPE  = gettype(observations)

    detector = observations.detector
    observations.psfs = zeros(FTYPE, build_dim, build_dim, observations.nsubaps, observations.nepochs, atmosphere.nλ)
    observations.images = zeros(FTYPE, observations.dim, observations.dim, observations.nsubaps, observations.nepochs)
    observations.monochromatic_images = zeros(FTYPE, observations.dim, observations.dim, observations.nsubaps, observations.nepochs, atmosphere.nλ)
    println("Creating $(observations.nepochs) images at $(observations.dim)x$(observations.dim) pixels for $(observations.nsubaps) subapertures across $(atmosphere.nλ) wavelengths")

    nthreads = Threads.nthreads()
    P = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
    p = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
    psf_temp = zeros(FTYPE, build_dim, build_dim, nthreads)
    image_temp_big = zeros(FTYPE, build_dim, build_dim, nthreads)
    iffts = [setup_ifft(build_dim, FTYPE=FTYPE) for tid=1:Threads.nthreads()]

    Threads.@threads :static for t=1:observations.nepochs
        tid = Threads.threadid()
        for n=1:observations.nsubaps
            for w=1:atmosphere.nλ
                refraction = create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations.ζ, detector.pixscale[w], build_dim, FTYPE=FTYPE)
                pupil2psf!(observations.psfs[:, :, n, t, w], psf_temp[:, :, tid], masks.masks[:, :, n, w], P[:, :, tid], p[:, :, tid], atmosphere.A[:, :, t, w], atmosphere.ϕ_composite[:, :, t, w], observations.α, masks.scale_psfs[w], iffts[tid], refraction)
            end
            create_polychromatic_image!(observations.images[:, :, n, t], observations.monochromatic_images[:, :, n, t, :], image_temp_big[:, :, tid], object.object, observations.psfs[:, :, n, t, :], atmosphere.λ, atmosphere.Δλ)
            if noise == true
                add_noise!(observations.images[:, :, n, t], observations.detector.rn, true, FTYPE=FTYPE)
            end
        end
    end
end
