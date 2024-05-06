mutable struct PatchHelpers{T<:AbstractFloat}
    extractor::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    extractor_adj::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    P::Array{Complex{T}, 5}
    p::Array{Complex{T}, 5}
    @views function PatchHelpers(patches, observations, atmosphere, object, reconstruction; FTYPE=Float64)
        P = zeros(Complex{FTYPE}, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, reconstruction.nλ, Threads.nthreads())
        p = zeros(Complex{FTYPE}, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, reconstruction.nλ, Threads.nthreads())

        Dmeta = observations.D .+ (object.fov/206265) .* (atmosphere.heights .* 1000)
        scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
        scaleby_height = Dmeta ./ observations.D
        extractor = create_patch_extractors(patches, atmosphere, observations, object, scaleby_wavelength, scaleby_height)
        extractor_adj = create_patch_extractors_adjoint(patches, atmosphere, observations, object, scaleby_wavelength, scaleby_height)
        return new{FTYPE}(extractor, extractor_adj, P, p)
    end
end

function reconstruct_blind_aniso!(reconstruction, observations, atmosphere, object, masks, patches, helpers, patch_helpers, regularizers; verb="mfbd", plot=false)
    FTYPE = gettype(reconstruction)
    if verb == "full"
        vm = true
        vo = true
    elseif verb == "mfbd"
        vm = true
        vo = false
    elseif verb == "silent"
        vm = false
        vo = false
    end

    if plot == true
        ## Plots the object and both phase layers
        fig = Figure(size=(600*atmosphere.nlayers, 600))
        o_ax = Axis(fig[1, 1])
        o_ax.title = "Object (mean)"
        l_ax = [Axis(fig[1, 2*l+1]) for l=1:atmosphere.nlayers]
        [l_ax[l].title = "Layer $(l) - $(reconstruction.λ[1])nm" for l=1:atmosphere.nlayers]
        [hidedecorations!(ax) for ax in l_ax]
        hidedecorations!(o_ax)
        o_ax.aspect = DataAspect()
        [ax.aspect = DataAspect() for ax in l_ax]
        global o_obs = Observable(rotr90(dropdims(mean(object.object, dims=3), dims=3)))
        global l_obs = [Observable(rotr90(atmosphere.ϕ[:, :, l, 1].*atmosphere.masks[:, :, l, 1])) for l=1:atmosphere.nlayers]
        o_plt = plot!(o_ax, o_obs, colormap=Reverse(:Greys))
        l_plt = [plot!(l_ax[l], l_obs[l], colormap=Reverse(:Greys)) for l=1:atmosphere.nlayers]
        Colorbar(fig[1, 2], o_plt, width=10, height=Relative(2/4))
        [Colorbar(fig[1, 2*l+2], l_plt[l], width=10, height=Relative(2/4)) for l=1:atmosphere.nlayers]
        trim!(fig.layout)
        display(fig)
    end

    nthreads = Threads.nthreads()
    k = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim)
    k_conv = Vector{typeof(plan_conv_psf_buffer(k, k)[2])}(undef, nthreads)
    k_corr = Vector{typeof(plan_ccorr_psf_buffer(k, Complex{FTYPE}.(k))[2])}(undef, nthreads)
    o_conv = Array{typeof(plan_conv_psf_buffer(k, k)[2]), 3}(undef, reconstruction.nλ, patches.npatches, nthreads)
    o_corr = Array{typeof(plan_ccorr_psf_buffer(k, k)[2]), 3}(undef, reconstruction.nλ, patches.npatches, nthreads)
    patches.psfs = Vector{Array{FTYPE, 6}}(undef, reconstruction.ndatasets)
    for dd=1:reconstruction.ndatasets
        patches.psfs[dd] = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].psfs = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].model_images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
    end

    ## Perform MFBD iterations
    for b=1:reconstruction.niter_boot
        ndatasets_current = length(reconstruction.indx_boot[b])
        current_observations = observations[reconstruction.indx_boot[b]]
        current_masks = masks[reconstruction.indx_boot[b]]
        
        fill!(reconstruction.ϵ, zero(FTYPE))
        for current_iter=1:reconstruction.niter_mfbd
            if vm == true
                print("Bootstrap Iter: $(b) MFBD Iter: $(current_iter) ")
            end
        
            regularizers.βo *= regularizers.βo_schedule(current_iter)
            regularizers.βϕ *= regularizers.βϕ_schedule(current_iter)
            
            fwhm = reconstruction.fwhm_schedule((b-1)*reconstruction.niter_boot + current_iter)
            k .= gaussian_kernel(reconstruction.build_dim, fwhm, FTYPE=FTYPE)
            for tid=1:nthreads
                k_conv[tid] = plan_conv_psf_buffer(k, k)[2]
                k_corr[tid] = plan_ccorr_psf_buffer(k, Complex{FTYPE}.(k))[2]
            end
            
            for w=1:reconstruction.nλ
                for np=1:patches.npatches
                    for tid=1:nthreads
                        o_conv[w, np, tid] = plan_conv_psf_buffer(k, patches.w[:, :, np] .* object.object[:, :, w])[2]
                        o_corr[w, np, tid] = plan_ccorr_psf_buffer(k, patches.w[:, :, np] .* object.object[:, :, w])[2]
                    end
                end
            end

            ## Reconstruct complex pupil
            if vm == true
                print("--> Reconstructing complex pupil ")
            end

            ## Reconstruct Phase
            crit_phase = (x, g) -> fg_phase(x, g, current_observations, atmosphere, current_masks, patches, reconstruction, helpers, patch_helpers, regularizers, o_conv, o_corr, k_conv, k_corr)
            vmlmb!(crit_phase, atmosphere.ϕ, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, fmin=0)
            GC.gc()

            if plot == true
                for l=1:atmosphere.nlayers
                    l_obs[l][] = rotr90(atmosphere.masks[:, :, l, 1] .* atmosphere.ϕ[:, :, l, 1])
                end
            end

            ## Reconstruct Object
            if vm == true
                print("--> object ")
            end

            crit_obj = (x, g) -> fg_object(x, g, current_observations, reconstruction, patches, helpers, regularizers)
            vmlmb!(crit_obj, object.object, lower=0, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, fmin=0)
            GC.gc()

            if plot == true
                o_obs[] = rotr90(dropdims(mean(object.object, dims=3), dims=3))
            end

            ## Compute final criterion
            reconstruction.ϵ[current_iter] = fg_object(object.object, similar(object.object), current_observations, reconstruction, patches, helpers, regularizers)

            if vm == true
                print("--> ϵ:\t$(reconstruction.ϵ[current_iter])\n")
            end
        end
    end
end

@views function object_solve_aniso(reconstruction, observations, object, patches, helpers, regularizers; verb="mfbd", plot=false)
    FTYPE = gettype(reconstruction)
    if verb == "full"
        vm = true
        vo = true
    elseif verb == "mfbd"
        vm = true
        vo = false
    elseif verb == "silent"
        vm = false
        vo = false
    end

    if plot == true
        ## Plots the object and both phase layers
        fig = Figure(size=(600, 600))
        o_ax = Axis(fig[1, 1])
        o_ax.title = "Object (mean)"
        hidedecorations!(o_ax)
        o_ax.aspect = DataAspect()
        global o_obs = Observable(rotr90(dropdims(mean(object.object, dims=3), dims=3)))
        o_plt = plot!(o_ax, o_obs, colormap=Reverse(:Greys))
        Colorbar(fig[1, 2], o_plt, width=10, height=Relative(2/4))
        trim!(fig.layout)
        display(fig)
    end
    fill!(reconstruction.ϵ, zero(FTYPE))    
    
    crit_obj = (x, g) -> fg_object(x, g, observations, reconstruction, patches, helpers, regularizers)
    vmlmb!(crit_obj, object.object, lower=0, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, fmin=0)
    GC.gc()
    
    reconstruction.ϵ[1] = fg_object(object.object, similar(object.object), observations, reconstruction, patches, helpers, regularizers)
end

# @views function phase_solve_aniso(reconstruction, observations, atmosphere, object, masks, patches, helpers, patch_helpers, regularizers; verb="mfbd", plot=false)

# end


@views function height_solve!(observations, atmosphere, object, patches, masks, reconstruction, regularizers, helpers, patch_helpers,; hmin=ones(atmosphere.nlayers-1), hmax=30.0.*ones(atmosphere.nlayers-1), hstep=ones(atmosphere.nlayers-1), niters=1, verb=true)
    if verb == true
        println("Solving heights for $(atmosphere.nlayers-1) layers")
    end
    
    FTYPE = gettype(reconstruction)
    nlayers2fit = atmosphere.nlayers - 1
    order2fit = reverse(sortperm(atmosphere.wind[:, 1])[2:end])
    heights = zeros(FTYPE, atmosphere.nlayers)
    heights .= atmosphere.heights
    ϵ = Vector{Vector{FTYPE}}(undef, nlayers2fit)
    height_trials = reverse([hmin[l]:hstep[l]:hmax[l] for l=1:nlayers2fit])
    atmosphere_original = deepcopy(atmosphere)
    object_original = deepcopy(object)

    nthreads = Threads.nthreads()
    psf_temp = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, nthreads)
    image_temp_big = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, nthreads)
    image_temp_small = [zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads) for dd=1:reconstruction.ndatasets]
    object_patch = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, nthreads)
    P = zeros(Complex{FTYPE}, reconstruction.build_dim, reconstruction.build_dim, nthreads)
    p = zeros(Complex{FTYPE}, reconstruction.build_dim, reconstruction.build_dim, nthreads)
    iffts = [setup_ifft(reconstruction.build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
    refraction = [[create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale[w], reconstruction.build_dim, FTYPE=FTYPE) for w=1:atmosphere.nλ] for dd=1:reconstruction.ndatasets]
    patches.A = ones(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[end].nepochs, atmosphere.nλ)
    patches.ϕ_composite = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[end].nepochs, atmosphere.nλ)
    patches.psfs = Vector{Array{FTYPE, 6}}(undef, reconstruction.ndatasets)
    for dd=1:reconstruction.ndatasets
        patches.psfs[dd] = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].psfs = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].model_images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
    end
    fill!(helpers.ϵ_threads, zero(FTYPE))

    for it=1:niters
        atmosphere = deepcopy(atmosphere_original)
        for l=1:nlayers2fit
            ϵ[l] = zeros(FTYPE, length(height_trials[l]))
            for h=1:length(height_trials[l])
                heights[order2fit[l]] = height_trials[l][h]
                print("\tHeight: $(heights)\t")
                change_heights!(patches, atmosphere, object, observations[end], masks[end], heights, helpers=helpers, patch_helpers=patch_helpers)
                calculate_composite_pupil_eff!(patches.A, patches.ϕ_composite, helpers.ϕ_slice, patches, atmosphere, observations[end], masks[end], patch_helpers.extractor, propagate=false)
                for dd=1:reconstruction.ndatasets
                    observation = observations[dd]
                    mask = masks[dd]
                    psfs = patches.psfs[dd]
                    r = helpers.r[dd]
                    ω = helpers.ω[dd]
                    image_small = image_temp_small[dd]
                    fill!(observation.model_images, zero(FTYPE))
                    Threads.@threads :static for n=1:observation.nsubaps
                        tid = Threads.threadid()
                        for t=1:observation.nepochs
                            for np=1:patches.npatches
                                for w=1:atmosphere.nλ
                                    pupil2psf!(psfs[:, :, np, n, t, w], psf_temp[:, :, tid], mask.masks[:, :, n, w], P[:, :, tid], p[:, :, tid], atmosphere.A[:, :, t, w], patches.ϕ_composite[:, :, np, t, w], observation.α, mask.scale_psfs[w], iffts[tid], refraction[dd][w])
                                end
                                create_polychromatic_image!(observation.model_images[:, :, n, t], image_small[:, :, tid], image_temp_big[:, :, tid], patches.w[:, :, np], object_patch[:, :, tid], object.object, psfs[:, :, np, n, t, :], atmosphere.λ, atmosphere.Δλ)
                            end
                            ω[:, :, tid] .= (observation.entropy[n, t] * observation.detector.rn^2)^(-1)
                            helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid])
                        end
                    end
                end
                ϵ[l][h] = sum(helpers.ϵ_threads)
                # object_solve_aniso(reconstruction, observations, object, patches, helpers, regularizers; verb="full", plot=true)
                # ϵ[l][h] = reconstruction.ϵ[1]
                println("ϵ: $(ϵ[l][h])")
                atmosphere = deepcopy(atmosphere_original)
                object = deepcopy(object_original)
            end
            heights[order2fit[l]] = height_trials[l][argmin(ϵ[l])]
            change_heights!(patches, atmosphere, object, observations[end], masks[end], heights, helpers=helpers, patch_helpers=patch_helpers, reconstruction=reconstruction)
            object_solve_aniso(reconstruction, observations, object, patches, helpers, regularizers; verb="full", plot=true)
            object_original = deepcopy(object)
        end
        println("Optimal Heights: $(heights)")
        change_heights!(patches, atmosphere, object, observations[end], masks[end], heights, helpers=helpers, patch_helpers=patch_helpers, reconstruction=reconstruction)
    end

    return ϵ, height_trials
end
