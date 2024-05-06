mutable struct AnisoplanaticPatches{T<:AbstractFloat}  # Structure to store isoplanatic patch info
    npatches::Int64  # Number of patches per image side length, e.g. 7x7 patches per images -> npatches_per_image_side = 7
    dim::Int64      # Patch width, e.g. 64x64 pixel patch -> npix_isopatch_width = 64
    overlap::T         # Percentage of window patch overlap, e.g. 50% overlap -> patch_overlap = 0.5
    coords::Matrix{Vector{Int64}}  # Patch lower and upper bounds
    positions::Matrix{T} # Locations of the patch centers, relative to the image center
    w::Array{T, 3}                  # Windowing function
    ϕ_slices::Array{T, 6}
    ϕ_composite::Array{T, 5}
    psfs::Vector{Array{T, 6}}
    @views function AnisoplanaticPatches(
            dim, 
            image_dim, 
            overlap; 
            isoplanatic=false,
            verb=true, 
            FTYPE=Float64, 
            ITYPE=Int64
        )
        if isoplanatic == true
            dim = image_dim
            overlap = 0
        end

        npatches_side = dim==image_dim ? 1 : ceil(ITYPE, image_dim / (dim * (1 - overlap))) + 1
        # npatches_side = round(ITYPE, image_dim / dim)

        if verb == true
            println("Creating $(npatches_side)×$(npatches_side) patches ($(dim) pix) across source for anisoplanatic model")
        end
        
        npatches = npatches_side^2
        xcenters = round.(ITYPE, [0:npatches_side-1;] .* dim .* (1-overlap) .+ 1)
        xstarts = xcenters .- (dim÷2)
        xends = xcenters .+ (dim÷2-1)
        positions = [[i for i in xcenters for j in xcenters] [j for i in xcenters for j in xcenters]]' .- (image_dim÷2+1);
        xpos = [[xstarts[n], xends[n]] for n=1:npatches_side];
        coords = hcat([i for i in xpos for j in xpos], [j for i in xpos for j in xpos]);
        w = zeros(FTYPE, image_dim, image_dim, npatches);
        xx = repeat([0:image_dim-1;], 1, image_dim); 
        yy = xx';
        mask = zeros(FTYPE, image_dim, image_dim)
        for n=1:npatches
            fill!(mask, zero(FTYPE))
            xrange = max(coords[n, 2][1], 1):min(coords[n, 2][2], image_dim)
            yrange = max(coords[n, 1][1], 1):min(coords[n, 1][2], image_dim)
            if isoplanatic == true
                w[:, :, n] .= 1
            else
                mask[yrange, xrange] .= 1
                w[:, :, n] .= mask .* bartlett_hann2d(xx .- (image_dim - dim)/2 .- positions[1, n], yy.- (image_dim - dim)/2 .- positions[2, n], dim)
            end
        end

        return new{FTYPE}(npatches, dim, overlap, coords, positions, w)
    end
end

function get_center(field_point, pupil_position, object_sampling_arcsecperpix, atmosphere_sampling_nyquist_mperpix, height, scaleby_wavelength)
    center = similar(field_point)
    get_center!(center, field_point, pupil_position, object_sampling_arcsecperpix, atmosphere_sampling_nyquist_mperpix, height, scaleby_wavelength)
    return center
end

function get_center!(center, field_point, pupil_position, object_sampling_arcsecperpix, atmosphere_sampling_nyquist_mperpix, height, scaleby_wavelength)
    center .= field_point  # Δx [pix in source plane]
    center .*= object_sampling_arcsecperpix / 206265  # Δx [radians]
    center .*= height*1000  # Δx [meters in layer]
    center ./= atmosphere_sampling_nyquist_mperpix  # Δx  [pix in layer]
    center .*= scaleby_wavelength
    center .+= pupil_position
end

@views function create_patch_extractors(patches, atmosphere, observations, object, scaleby_wavelength, scaleby_height; build_dim=256)
    FTYPE = gettype(atmosphere)
    extractor = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, observations.nepochs, patches.npatches, atmosphere.nlayers, atmosphere.nλ)
    Threads.@threads for t=1:observations.nepochs
        for n=1:patches.npatches
            for w=1:atmosphere.nλ
                for l=1:atmosphere.nlayers
                    center = get_center(patches.positions[:, n], atmosphere.positions[:, t, l, w], object.sampling_arcsecperpix, atmosphere.sampling_nyquist_mperpix[l], atmosphere.heights[l], scaleby_wavelength[w])
                    extractor[t, n, l, w] = create_extractor_operator(center, atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                end
            end
        end
    end

    return extractor
end

@views function create_patch_extractors_adjoint(patches, atmosphere, observations, object, scaleby_wavelength, scaleby_height; build_dim=256)
    FTYPE = gettype(atmosphere)
    extractor_adj = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, observations.nepochs, patches.npatches, atmosphere.nlayers, atmosphere.nλ)
    Threads.@threads for t=1:observations.nepochs
        for n=1:patches.npatches
            for w=1:atmosphere.nλ
                for l=1:atmosphere.nlayers
                    center = get_center(patches.positions[:, n], atmosphere.positions[:, t, l, w], object.sampling_arcsecperpix, atmosphere.sampling_nyquist_mperpix[l], atmosphere.heights[l], scaleby_wavelength[w])
                    extractor_adj[t, n, l, w] = create_extractor_adjoint(center, atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                end
            end
        end
    end

    return extractor_adj
end

@views function calculate_layer_masks_eff!(patches, atmosphere, observations, object, masks)
    FTYPE = gettype(atmosphere)
    Dmeta = observations[end].D .+ (object.fov/206265) .* (atmosphere.heights .* 1000)
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    scaleby_height = Dmeta ./ observations[end].D
    buffer = zeros(FTYPE, atmosphere.dim, atmosphere.dim, Threads.nthreads())
    layer_mask = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ, Threads.nthreads())
    deextractors = create_patch_extractors_adjoint(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
    ndatasets = length(observations)
    for dd=1:ndatasets
        observation = observations[dd]
        mask = masks[dd]
        Threads.@threads :static for t=1:observation.nepochs
            tid = Threads.threadid()
            for np=1:patches.npatches
                for n=1:observation.nsubaps
                    for w=1:atmosphere.nλ
                        for l=1:atmosphere.nlayers
                            position2phase!(buffer[:, :, tid], mask.masks[:, :, n, w], deextractors[t, np, l, w])
                            layer_mask[:, :, l, w, tid] .+= buffer[:, :, tid]
                        end
                    end
                end
            end
        end
    end
    atmosphere.masks = dropdims(sum(layer_mask, dims=5), dims=5)
    atmosphere.masks[atmosphere.masks .> 0] .= 1
end

@views function calculate_layer_masks_eff_alt!(atmosphere, observations, object, masks)
    FTYPE = gettype(atmosphere)
    Dmeta = observations.D .+ (object.fov/206265) .* (atmosphere.heights .* 1000)
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    scaleby_height = Dmeta ./ observations.D
    calculate_layer_masks_iso!(atmosphere, observations, object, masks)
    for w=1:atmosphere.nλ
        for l=1:atmosphere.nlayers
            enlarger = create_extractor_operator((atmosphere.dim÷2, atmosphere.dim÷2), atmosphere.dim, atmosphere.dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
            atmosphere.masks[:, :, l, w] .= enlarger * atmosphere.masks[:, :, l, w]
        end
    end
end

@views function calculate_composite_pupil_eff(patches, atmosphere, observations, object, masks; build_dim=256, propagate=true, verb=true)
    if verb == true
        println("Calculating composite complex pupil for $(length(observations)) channels")
    end
    FTYPE = gettype(atmosphere)
    [observations[dd].A = ones(FTYPE, build_dim, build_dim, observations[dd].nsubaps, patches.npatches, observations[dd].nepochs, atmosphere.nλ) for dd=1:length(observations)]
    patches.ϕ_slices = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[end].nepochs, atmosphere.nlayers, atmosphere.nλ)
    patches.ϕ_composite = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[end].nepochs, atmosphere.nλ)

    Dmeta = observations[end].D .+ (object.fov/206265) .* (atmosphere.heights .* 1000)
    scaleby_height = Dmeta ./ observations[end].D
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    extractors = create_patch_extractors(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height, build_dim=build_dim)
    if verb == true
        println("\tExtracting composite phase")
    end
    calculate_composite_phase_eff!(patches.ϕ_composite, patches.ϕ_slices, patches, atmosphere, observations[end], extractors)
    if propagate == true
        for dd=1:length(observations)
            if verb == true
                println("\tExtracting composite amplitude for channel $(dd) by Fresnel propagation")
            end
            calculate_composite_amplitude_eff!(observations[dd].A, masks[dd], observations[dd], atmosphere, patches)
        end
    end
end

@views function calculate_composite_phase_eff!(ϕ_composite, ϕ_slices, patches, atmosphere, observations, extractors)
    FTYPE = gettype(atmosphere)
    fill!(ϕ_composite, zero(FTYPE))
    Threads.@threads for t=1:observations.nepochs
        for np=1:patches.npatches
            for w=1:atmosphere.nλ
                for l=1:atmosphere.nlayers
                    position2phase!(ϕ_slices[:, :, np, t, l, w], FTYPE(2pi) ./ atmosphere.λ[w] .* atmosphere.opd[:, :, l], extractors[t, np, l, w])
                    ϕ_composite[:, :, np, t, w] .+= ϕ_slices[:, :, np, t, l, w]
                end
            end
        end
    end
end

@views function calculate_composite_amplitude_eff!(A, masks, observations, atmosphere, patches)
    FTYPE = gettype(atmosphere)
    Threads.@threads for t=1:observations.nepochs
        for np=1:patches.npatches
            for w=1:atmosphere.nλ
                for n=1:observations.nsubaps
                    ϕ_slices_subap = repeat(masks.masks[:, :, n, w], 1, 1, atmosphere.nlayers) .* patches.ϕ_slices[:, :, np, t, :, w]
                    N = size(patches.ϕ_slices, 1)
                    x1 = ((-N÷2:N÷2-1) .* atmosphere.sampling_nyquist_mperpix[1])' .* ones(N)
                    y1 = x1'
                    sg = repeat(exp.(-(x1 ./ (0.47*N)).^16) .* exp.(-(y1 ./ (0.47*N)).^16), 1, 1, atmosphere.nlayers)
                    Uout = propagate_layers(ones(Complex{FTYPE}, size(masks.masks[:, :, n, w])), atmosphere.λ[w], atmosphere.sampling_nyquist_mperpix[1], atmosphere.sampling_nyquist_mperpix[end], atmosphere.heights, sg .* cis.(ϕ_slices_subap))
                    A[:, :, n, np, t, w] .= masks.masks[:, :, n, w] .* abs.(Uout)
                end
            end
        end
    end
end

@views function create_images_eff(patches, observations, atmosphere, masks, object; build_dim=256, noise=false, verb=true)
    if verb == true
        println("Creating $(build_dim)×$(build_dim) images for $(observations.nepochs) times and $(observations.nsubaps) subaps using eff model")
    end
    FTYPE = gettype(patches)
    nthreads = Threads.nthreads()
    psf_temp = zeros(FTYPE, build_dim, build_dim, nthreads)
    image_temp_big = zeros(FTYPE, build_dim, build_dim, nthreads)
    image_temp_small = zeros(FTYPE, observations.dim, observations.dim, nthreads)
    object_patch = zeros(FTYPE, build_dim, build_dim, nthreads)
    nthreads = Threads.nthreads()
    P = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
    p = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
    iffts = [setup_ifft(build_dim, FTYPE=FTYPE) for tid=1:Threads.nthreads()]
    psfs = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations.nsubaps, observations.nepochs, atmosphere.nλ)
    images = zeros(FTYPE, observations.dim, observations.dim, observations.nsubaps, observations.nepochs)
    refraction = [create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations.ζ, observations.detector.pixscale[w], build_dim, FTYPE=FTYPE) for w=1:atmosphere.nλ]
    create_images_eff!(images, patches, observations, atmosphere, masks, object, object_patch, psfs, psf_temp, image_temp_small, image_temp_big, P, p, iffts, refraction; build_dim=build_dim, noise=noise)
    return images, psfs
end

@views function create_images_eff!(images, patches, observations, atmosphere, masks, object, object_patch, psfs, psf_temp, image_temp_small, image_temp_big, P, p, iffts, refraction; build_dim=256, noise=false)
    FTYPE = gettype(patches)
    Threads.@threads :static for t=1:observations.nepochs
        tid = Threads.threadid()
        for n=1:observations.nsubaps
            for np=1:patches.npatches
                for w=1:atmosphere.nλ
                    pupil2psf!(psfs[:, :, np, n, t, w], psf_temp[:, :, tid], masks.masks[:, :, n, w], P[:, :, tid], p[:, :, tid], observations.A[:, :, n, np, t, w], patches.ϕ_composite[:, :, np, t, w], observations.α, masks.scale_psfs[w], iffts[tid], refraction[w])
                end
                create_polychromatic_image!(images[:, :, n, t], image_temp_small[:, :, tid], image_temp_big[:, :, tid], patches.w[:, :, np], object_patch[:, :, tid], object.object, psfs[:, :, np, n, t, :], atmosphere.λ, atmosphere.Δλ)
            end
            if noise == true
                add_noise!(images[:, :, n, t], observations.detector.rn, true, FTYPE=FTYPE)
            end
        end
    end
end

# @views function create_images_raytrace(observations, atmosphere, masks, object; build_dim=256, noise=false, verb=true)
#     FTYPE = gettype(observations)
#     images = zeros(FTYPE, observations.dim, observations.dim, observations.nsubaps, observations.nepochs)
#     create_images_raytrace!(images, observations, atmosphere, masks, object, build_dim=build_dim, noise=noise, verb=verb)
#     return images
# end

# @views function create_images_raytrace!(images, observations, atmosphere, masks, object; build_dim=256, noise=false, verb=true)
#     if verb == true
#         println("Performing raytrace for $(observations.nsubaps) subaps at $(observations.nepochs) epochs and $(atmosphere.nλ) wavelengths")
#     end
        
#     FTYPE = gettype(observations)

#     nthreads = Threads.nthreads()
    
#     ϕ_composite = zeros(FTYPE, build_dim, build_dim, nthreads)
#     ϕ_slice = zeros(FTYPE, build_dim, build_dim, nthreads)
    
#     P = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
#     p = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
#     psf = zeros(FTYPE, build_dim, build_dim, nthreads)
#     psf_temp = zeros(FTYPE, build_dim, build_dim, nthreads)

#     object_pix = zeros(FTYPE, build_dim, build_dim, nthreads)
#     image_big = zeros(FTYPE, build_dim, build_dim, nthreads)
#     image_small = zeros(FTYPE, observations.dim, observations.dim, nthreads)
    
#     iffts = [setup_ifft(build_dim, FTYPE=FTYPE) for tid=1:Threads.nthreads()]    
#     scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
#     scaleby_height = 1 .- atmosphere.heights ./ object.height

#     if verb == true
#         prog = Progress(observations.nepochs*observations.nsubaps*atmosphere.nλ)
#     end
#     Threads.@threads :static for t=1:observations.nepochs
#         tid = Threads.threadid()
#         for n=1:observations.nsubaps
#             for w=1:atmosphere.nλ
#                 nonzero_pixels = findall(object.object[:, :, w] .> 0)
#                 refraction = create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations.ζ, observations.detector.pixscale[w], build_dim, FTYPE=FTYPE)
#                 for nzp in nonzero_pixels
#                     fill!(ϕ_composite[:, :, tid], zero(FTYPE))
#                     for l=1:atmosphere.nlayers
#                         Δpix = ((Tuple(nzp) .- (build_dim÷2+1, build_dim÷2+1)) .* object.sampling_arcsecperpix) ./ (atmosphere.sampling_nyquist_arcsecperpix[l] * (atmosphere.λ[w] / atmosphere.λ_nyquist))
#                         center = Tuple(atmosphere.positions[:, t, l, w] .+ Δpix)
#                         extractor = create_extractor_operator(center, atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
#                         mul!(ϕ_slice[:, :, tid], extractor, atmosphere.ϕ[:, :, l, w])
#                         ϕ_composite[:, :, tid] .+= ϕ_slice[:, :, tid]
#                     end
#                     pupil2psf!(psf[:, :, tid], psf_temp[:, :, tid], masks.masks[:, :, n, w], P[:, :, tid], p[:, :, tid], atmosphere.A[:, :, t, w], ϕ_composite[:, :, tid], observations.α, masks.scale_psfs[w], iffts[tid], refraction)
                    
#                     object_pix[nzp, tid] = object.object[nzp, w]
#                     create_monochromatic_image!(image_small[:, :, tid], image_big[:, :, tid], object_pix[:, :, tid], psf[:, :, tid])
#                     images[:, :, n, t] .+= image_small[:, :, tid]
#                     object_pix[nzp, tid] = zero(FTYPE)
#                 end
#                 if noise == true
#                     add_noise!(images[:, :, n, t], observations.detector.rn, true, FTYPE=FTYPE)
#                 end
#                 if verb == true
#                     next!(prog)
#                 end
#             end
#         end
#     end
#     if verb == true
#         finish!(prog)
#     end
# end

@views function change_heights!(patches, atmosphere, object, observations, masks, heights; reconstruction=[])
    FTYPE = gettype(observations[end])
    original_heights = atmosphere.heights
    Dmeta0 = observations[end].D .+ (object.fov/206265) .* (original_heights .* 1000) 
    Dmeta = observations[end].D .+ (object.fov/206265) .* (heights .* 1000)
    atmosphere.sampling_nyquist_mperpix = (2*observations[end].D / observations[end].dim) .* ones(atmosphere.nlayers)

    atmosphere.heights = heights
    calculate_screen_size!(atmosphere, observations[end], object, patches)
    calculate_pupil_positions!(atmosphere, observations[end])
    opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
    calculate_layer_masks_eff_alt!(atmosphere, observations[end], object, masks[end])

    if reconstruction != []
        helpers = reconstruction.helpers
        patch_helpers = reconstruction.patch_helpers
        if helpers != []
            helpers = reconstruction.helpers
            helpers.g_threads_ϕ = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ, Threads.nthreads())
            helpers.g_threads_opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, Threads.nthreads())
            helpers.ϕ_full = zeros(FTYPE, atmosphere.dim, atmosphere.dim, Threads.nthreads())
            helpers.containers_sdim_real = zeros(FTYPE, atmosphere.dim, atmosphere.dim, Threads.nthreads())
        end

        if patch_helpers != []
            scaleby_height = Dmeta ./ observations[end].D
            scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
            patch_helpers.extractor = create_patch_extractors(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
            patch_helpers.extractor_adj = create_patch_extractors_adjoint(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
        end
    end

    scaleby_height = Dmeta ./ Dmeta0
    for l=1:atmosphere.nlayers
        kernel = LinearSpline(FTYPE)
        transform = AffineTransform2D{FTYPE}()
        screen_size = size(atmosphere.opd[:, :, l])
        output_size = (Int64(atmosphere.dim), Int64(atmosphere.dim))
        full_transform = ((transform + (screen_size[1]÷2+1, screen_size[2]÷2+1)) * (1/scaleby_height[l])) - (atmosphere.dim÷2+1, atmosphere.dim÷2+1)
        scaler = TwoDimensionalTransformInterpolator(output_size, screen_size, kernel, full_transform)
        mul!(opd[:, :, l], scaler, atmosphere.opd[:, :, l])
    end
    atmosphere.opd = opd
    atmosphere.opd .*= atmosphere.masks[:, :, :, 1]
end
