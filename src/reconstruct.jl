using FFTW
FFTW.set_num_threads(1)
using Statistics
using FourierTools
using LinearAlgebra
using OptimPackNextGen


mutable struct Helpers{T<:AbstractFloat}
    extractor::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    extractor_adj::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    # extractor_adj::Array{LazyAlgebra.Adjoint{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}, 4}
    refraction::Matrix{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}
    refraction_adj::Matrix{LazyAlgebra.Adjoint{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}}
    ift::Vector{Function}
    W_smooth::Array{T, 3}
    ϕ_full::Array{T, 3}
    ϕ_slice::Array{T, 3}
    ϕ_composite::Array{T, 3}
    P::Array{Complex{T}, 4}
    p::Array{Complex{T}, 4}
    c::Array{T, 3}
    d::Array{Complex{T}, 3}
    d2::Array{T, 3}
    Î_big::Array{T, 3}
    Î_small::Vector{Array{T, 3}}
    r::Vector{Array{T, 3}}
    ω::Vector{Array{T, 3}}
    ϵ_threads::Vector{T}
    g_threads_ϕ::Array{T, 5}
    g_threads_obj::Array{T, 4}
    g_threads_opd::Array{T, 4}
    containers_builddim_real::Array{T, 3}
    containers_builddim_cplx::Array{Complex{T}, 3}
    containers_sdim_real::Array{T, 3}
    containers_sdim_cplx::Array{T, 3}
    containers_pdim_real::Vector{Array{T, 3}}
    containers_pdim_cplx::Vector{Array{Complex{T}, 3}}
    @views function Helpers(atmosphere,
                     observations,
                     object;
                     build_dim=size(object.object, 1),
                     ndatasets=length(observations),
                     verb=true,
                     FTYPE=gettype(atmosphere))
        if verb == true
            println("Setting up interpolators, buffers, fft/ifft plans, convolution/correlation plans")
        end
        nλ = atmosphere.nλ

        ifft_threads = [setup_ifft(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        extractor = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, ndatasets, observations[1].nepochs, atmosphere.nlayers, nλ)
        extractor_adj = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, ndatasets, observations[1].nepochs, atmosphere.nlayers, nλ)
        # extractor_adj = Array{LazyAlgebra.Adjoint{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}, 4}(undef, ndatasets, observations[1].nepochs, atmosphere.nlayers, nλ)
        refraction = Matrix{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}(undef, ndatasets, nλ)
        refraction_adj = Matrix{LazyAlgebra.Adjoint{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}}(undef, ndatasets, nλ)
        
        nthreads = Threads.nthreads()
        ϵ_threads = zeros(FTYPE, nthreads)
        g_threads_ϕ = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ, nthreads)
        g_threads_opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, nthreads)
        g_threads_obj = zeros(FTYPE, build_dim, build_dim, nλ, nthreads)
        W_smooth = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
        ϕ_full = zeros(FTYPE, atmosphere.dim, atmosphere.dim, nthreads)
        ϕ_slice = Array{FTYPE, 3}(undef, build_dim, build_dim, nthreads)
        ϕ_composite = Array{FTYPE, 3}(undef, build_dim, build_dim, nthreads)
        P = Array{Complex{FTYPE}, 4}(undef, build_dim, build_dim, nλ, nthreads)
        p = Array{Complex{FTYPE}, 4}(undef, build_dim, build_dim, nλ, nthreads)
        Î_big = Array{FTYPE, 3}(undef, build_dim, build_dim, nthreads)
        c = Array{FTYPE, 3}(undef, build_dim, build_dim, nthreads)
        d = Array{Complex{FTYPE}, 3}(undef, build_dim, build_dim, nthreads)
        d2 = Array{FTYPE, 3}(undef, build_dim, build_dim, nthreads)
        containers_builddim_cplx = Array{Complex{FTYPE}, 3}(undef, build_dim, build_dim, nthreads)
        containers_builddim_real = Array{FTYPE, 3}(undef, build_dim, build_dim, nthreads)
        containers_sdim_real = Array{FTYPE, 3}(undef, atmosphere.dim, atmosphere.dim, nthreads)
        containers_sdim_cplx = Array{Complex{FTYPE}, 3}(undef, atmosphere.dim, atmosphere.dim, nthreads)

        ndatasets = length(observations)
        r = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        ω = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        Î_small = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        containers_pdim_real = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        containers_pdim_cplx = Vector{Array{Complex{FTYPE}, 3}}(undef, ndatasets)
        
        Threads.@threads :static for dd=1:ndatasets
            r[dd] = Array{FTYPE, 3}(undef, observations[dd].dim, observations[dd].dim, nthreads)
            ω[dd] = Array{FTYPE, 3}(undef, observations[dd].dim, observations[dd].dim, nthreads)
            Î_small[dd] = Array{FTYPE, 3}(undef, observations[dd].dim, observations[dd].dim, nthreads)
            containers_pdim_real[dd] = Array{FTYPE, 3}(undef, observations[dd].dim, observations[dd].dim, nthreads)
            containers_pdim_cplx[dd] = Array{Complex{FTYPE}, 3}(undef, observations[dd].dim, observations[dd].dim, nthreads)
            scaleby_wavelength = [observations[dd].detector.λ_nyquist/atmosphere.λ[w] for w=1:atmosphere.nλ]
            scaleby_height = 1 .- atmosphere.heights ./ object.height
            for w=1:nλ
                refraction[dd, w] = create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale[w], build_dim, FTYPE=FTYPE)
                refraction_adj[dd, w] = refraction[dd, w]'
                for l=1:atmosphere.nlayers
                    for t=1:observations[dd].nepochs
                        extractor[dd, t, l, w] = create_extractor_operator(atmosphere.positions[:, t, l, w], atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                        extractor_adj[dd, t, l, w] = create_extractor_adjoint(atmosphere.positions[:, t, l, w], atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                    end
                end
            end
        end

        return new{FTYPE}(extractor, extractor_adj, refraction, refraction_adj, ifft_threads, W_smooth, ϕ_full, ϕ_slice, ϕ_composite, P, p, c, d, d2, Î_big, Î_small, r, ω, ϵ_threads, g_threads_ϕ, g_threads_obj, g_threads_opd, containers_builddim_real, containers_builddim_cplx, containers_sdim_real, containers_sdim_cplx, containers_pdim_real, containers_pdim_cplx)
    end
end

mutable struct PatchHelpers{T<:AbstractFloat}
    extractor::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    extractor_adj::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    P::Array{Complex{T}, 5}
    p::Array{Complex{T}, 5}
    @views function PatchHelpers(patches, observations, atmosphere, object; build_dim=size(object.object, 1), FTYPE=gettype(atmosphere))
        nλ = atmosphere.nλ
        P = zeros(Complex{FTYPE}, build_dim, build_dim, patches.npatches, nλ, Threads.nthreads())
        p = zeros(Complex{FTYPE}, build_dim, build_dim, patches.npatches, nλ, Threads.nthreads())

        Dmeta = observations[end].D .+ (object.fov/206265) .* (atmosphere.heights .* 1000)
        scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
        scaleby_height = Dmeta ./ observations[end].D
        extractor = create_patch_extractors(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
        extractor_adj = create_patch_extractors_adjoint(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
        return new{FTYPE}(extractor, extractor_adj, P, p)
    end
end

mutable struct Reconstruction{T<:AbstractFloat}
    λ::Vector{T}
    nλ::Int64
    Δλ::T
    ndatasets::Int64
    build_dim::Int64
    niter_mfbd::Int64
    indx_boot::Vector{UnitRange{Int64}}
    niter_boot::Int64
    maxiter::Int64
    ϵ::Vector{T}
    gtol::Tuple{T, T}
    xtol::Tuple{T, T}
    ftol::Tuple{T, T}
    regularizers::Regularizers{T}
    smoothing::Bool
    fwhm_schedule::Function
    helpers::Helpers{T}
    patch_helpers::PatchHelpers{T}
    function Reconstruction(
            atmosphere,
            observations,
            object,
            patches;
            λ=atmosphere.λ,
            ndatasets=length(observations),
            build_dim=size(object.object, 1),
            niter_mfbd=10,
            indx_boot=[1:dd for dd=1:ndatasets],
            maxiter=10,
            gtol=(0.0, 1e-9),
            xtol=(0.0, 1e-9),
            ftol=(0.0, 1e-9),
            smoothing=false,
            fwhm_schedule=x -> max(10 * (1 - x/niter_mfbd), 1.0),
            regularizers=[],
            helpers=[],
            patch_helpers=[],
            verb=true,
            FTYPE = gettype(atmosphere)
        )
        nλ = length(λ)
        Δλ = (nλ == 1) ? 1.0 : (maximum(λ) - minimum(λ)) / (nλ - 1)

        niter_boot = length(indx_boot)
        ϵ = zeros(FTYPE, niter_mfbd)
        if verb == true
            println("Setting up MFBD: $(nλ) wavelengths, $(niter_mfbd) iterations, $(ndatasets) aperture diverse channels")
        end

        if regularizers == []
            regularizers = Regularizers(FTYPE=FTYPE)
        end

        if helpers == []
            helpers = Helpers(
                atmosphere, 
                observations,
                object
            );
        end

        if patch_helpers == []
            patch_helpers = PatchHelpers(
                patches,
                observations,
                atmosphere,
                object
            )
        end

        return new{FTYPE}(λ, nλ, Δλ, ndatasets, build_dim, niter_mfbd, indx_boot, niter_boot, maxiter, ϵ, gtol, xtol, ftol, regularizers, smoothing, fwhm_schedule, helpers, patch_helpers)
    end
end

@views function object_solve!(reconstruction, observations, object, patches; verb="mfbd", plot=false)
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
        Colorbar(fig[1, 2], o_plt, width=10, height=Relative(1))
        trim!(fig.layout)
        display(fig)
    end
    fill!(reconstruction.ϵ, zero(FTYPE))    
    
    crit_obj = (x, g) -> fg_object(x, g, observations, reconstruction, patches)
    vmlmb!(crit_obj, object.object, lower=0, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, xtol=reconstruction.xtol, ftol=reconstruction.ftol, fmin=0)
    GC.gc()
    
    reconstruction.ϵ[1] = fg_object(object.object, similar(object.object), observations, reconstruction, patches)
end

@views function opd_solve!(reconstruction, observations, atmosphere, object, masks, patches; verb="mfbd", plot=false)
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
        l_ax = [Axis(fig[1, 2*l]) for l=1:atmosphere.nlayers]
        [l_ax[l].title = "Layer $(l) - OPD [nm]" for l=1:atmosphere.nlayers]
        [hidedecorations!(ax) for ax in l_ax]
        [ax.aspect = DataAspect() for ax in l_ax]
        global l_obs = [Observable(rotr90(atmosphere.opd[:, :, l].*atmosphere.masks[:, :, l, 1])) for l=1:atmosphere.nlayers]
        l_plt = [plot!(l_ax[l], l_obs[l], colormap=Reverse(:Greys)) for l=1:atmosphere.nlayers]
        [Colorbar(fig[1, 2*l+1], l_plt[l], width=10, height=Relative(2/4)) for l=1:atmosphere.nlayers]
        trim!(fig.layout)
        display(fig)
    end

    nthreads = Threads.nthreads()
    k = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim)
    k_conv = Vector{typeof(plan_conv_psf_buffer(k, k)[2])}(undef, nthreads)
    k_corr = Vector{typeof(plan_ccorr_psf_buffer(k, k)[2])}(undef, nthreads)
    o_conv = Array{typeof(plan_conv_psf_buffer(k, k)[2]), 3}(undef, reconstruction.nλ, patches.npatches, nthreads)
    o_corr = Array{typeof(plan_ccorr_psf_buffer(k, k)[2]), 3}(undef, reconstruction.nλ, patches.npatches, nthreads)
    patches.psfs = Vector{Array{FTYPE, 6}}(undef, reconstruction.ndatasets)
    for dd=1:reconstruction.ndatasets
        patches.psfs[dd] = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].psfs = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].model_images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
    end

    ## Perform MFBD iterations  
    fill!(reconstruction.ϵ, zero(FTYPE))

    k .= gaussian_kernel(reconstruction.build_dim, 0.5, FTYPE=FTYPE)
    for tid=1:nthreads
        k_conv[tid] = plan_conv_psf_buffer(k, k)[2]
        k_corr[tid] = plan_ccorr_psf_buffer(k, k)[2]
    end
    
    for w=1:reconstruction.nλ
        for np=1:patches.npatches
            for tid=1:nthreads
                o_conv[w, np, tid] = plan_conv_psf_buffer(object.object[:, :, w], patches.w[:, :, np] .* object.object[:, :, w])[2]
                o_corr[w, np, tid] = plan_ccorr_psf_buffer(object.object[:, :, w], patches.w[:, :, np] .* object.object[:, :, w])[2]
            end
        end
    end

    ## Reconstruct Phase
    crit_opd = (x, g) -> fg_opd(x, g, observations, atmosphere, masks, patches, reconstruction, o_conv, o_corr, k_conv, k_corr)
    vmlmb!(crit_opd, atmosphere.opd, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, xtol=reconstruction.xtol, ftol=reconstruction.ftol, fmin=0)
    GC.gc()

    reconstruction.ϵ[1] = fg_object(object.object, similar(object.object), observations, reconstruction, patches)
end

function reconstruct_blind!(reconstruction, observations, atmosphere, object, masks, patches; verb="mfbd", plot=false)
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
        [l_ax[l].title = "Layer $(l) - OPD [nm]" for l=1:atmosphere.nlayers]
        [hidedecorations!(ax) for ax in l_ax]
        hidedecorations!(o_ax)
        o_ax.aspect = DataAspect()
        [ax.aspect = DataAspect() for ax in l_ax]
        global o_obs = Observable(rotr90(dropdims(mean(object.object, dims=3), dims=3)))
        global l_obs = [Observable(rotr90(atmosphere.opd[:, :, l].*atmosphere.masks[:, :, l, 1])) for l=1:atmosphere.nlayers]
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
    k_corr = Vector{typeof(plan_ccorr_psf_buffer(k, k)[2])}(undef, nthreads)
    o_conv = Array{typeof(plan_conv_psf_buffer(k, k)[2]), 3}(undef, reconstruction.nλ, patches.npatches, nthreads)
    o_corr = Array{typeof(plan_ccorr_psf_buffer(k, k)[2]), 3}(undef, reconstruction.nλ, patches.npatches, nthreads)
    patches.psfs = Vector{Array{FTYPE, 6}}(undef, reconstruction.ndatasets)
    patches.ϕ_composite = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[1].nepochs, reconstruction.nλ)
    patches.ϕ_slices = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[1].nepochs, atmosphere.nlayers, reconstruction.nλ)
    for dd=1:reconstruction.ndatasets
        patches.psfs[dd] = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].psfs = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].model_images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
    end
    regularizers = reconstruction.regularizers

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
            # fwhm = reconstruction.fwhm_schedule(current_iter)
            k .= gaussian_kernel(reconstruction.build_dim, fwhm, FTYPE=FTYPE)
            for tid=1:nthreads
                k_conv[tid] = plan_conv_psf_buffer(k, k)[2]
                k_corr[tid] = plan_ccorr_psf_buffer(k, k)[2]
            end
            
            for w=1:reconstruction.nλ
                for np=1:patches.npatches
                    for tid=1:nthreads
                        o_conv[w, np, tid] = plan_conv_psf_buffer(object.object[:, :, w], patches.w[:, :, np] .* object.object[:, :, w])[2]
                        o_corr[w, np, tid] = plan_ccorr_psf_buffer(object.object[:, :, w], patches.w[:, :, np] .* object.object[:, :, w])[2]
                    end
                end
            end

            ## Reconstruct complex pupil
            if vm == true
                print("--> Reconstructing complex pupil ")
            end

            ## Reconstruct Phase
            crit_opd = (x, g) -> fg_opd(x, g, current_observations, atmosphere, current_masks, patches, reconstruction, o_conv, o_corr, k_conv, k_corr)
            vmlmb!(crit_opd, atmosphere.opd, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, xtol=reconstruction.xtol, ftol=reconstruction.ftol, fmin=0)
            GC.gc()

            if plot == true
                for l=1:atmosphere.nlayers
                    l_obs[l][] = rotr90(atmosphere.masks[:, :, l, 1] .* atmosphere.opd[:, :, l])
                end
            end

            ## Reconstruct Object
            if vm == true
                print("--> object ")
            end

            crit_obj = (x, g) -> fg_object(x, g, current_observations, reconstruction, patches)
            vmlmb!(crit_obj, object.object, lower=0, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, xtol=reconstruction.xtol, ftol=reconstruction.ftol, fmin=0)
            GC.gc()

            if plot == true
                o_obs[] = rotr90(dropdims(mean(object.object, dims=3), dims=3))
            end

            ## Compute final criterion
            reconstruction.ϵ[current_iter] = fg_object(object.object, similar(object.object), current_observations, reconstruction, patches)

            if vm == true
                print("--> ϵ:\t$(reconstruction.ϵ[current_iter])\n")
            end
        end
    end
end

@views function height_solve!(observations, atmosphere, object, patches, masks, reconstruction; hmin=ones(atmosphere.nlayers-1), hmax=30.0.*ones(atmosphere.nlayers-1), hstep=ones(atmosphere.nlayers-1), niters=1, verb=true)
    if verb == true
        println("Solving heights for $(atmosphere.nlayers-1) layers")
    end

    FTYPE = gettype(reconstruction)
    nlayers2fit = atmosphere.nlayers - 1
    order2fit = reverse(sortperm(atmosphere.wind[:, 1])[2:end])
    heights = zeros(FTYPE, atmosphere.nlayers)
    heights .= atmosphere.heights
    ϵ = Vector{Vector{FTYPE}}(undef, nlayers2fit)
    height_trials = reverse([collect(hmin[l]:hstep[l]:hmax[l]) for l=1:nlayers2fit])
    # [heights[order2fit[l]] = height_trials[l][1] for l=1:nlayers2fit]
    atmosphere_original = deepcopy(atmosphere)
    object_original = deepcopy(object)

    patches.ϕ_slices = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[end].nepochs, atmosphere.nlayers, atmosphere.nλ)
    patches.ϕ_composite = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[end].nepochs, atmosphere.nλ)
    patches.psfs = Vector{Array{FTYPE, 6}}(undef, reconstruction.ndatasets)
    for dd=1:reconstruction.ndatasets
        patches.psfs[dd] = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].A = ones(FTYPE, reconstruction.build_dim, reconstruction.build_dim, observations[dd].nsubaps, patches.npatches, observations[dd].nepochs, atmosphere.nλ)
        observations[dd].psfs = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, observations[dd].nsubaps, observations[dd].nepochs, reconstruction.nλ)
        observations[dd].model_images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
    end
    
    for it=1:niters
        atmosphere = deepcopy(atmosphere_original)
        for l=1:nlayers2fit
            ϵ[l] = zeros(FTYPE, length(height_trials[l]))
            for h=1:length(height_trials[l])
                heights[order2fit[l]] = height_trials[l][h]
                print("\tHeight: $(heights)\t")
                change_heights!(patches, atmosphere, object, observations, masks, heights, reconstruction=reconstruction)
                opd_solve!(reconstruction, observations, atmosphere, object, masks, patches, verb="full", plot=false)
                object_solve!(reconstruction, observations, object, patches; verb="full", plot=false)
                ϵ[l][h] = sum(reconstruction.ϵ[1])
                println("ϵ: $(ϵ[l][h])")
                atmosphere = deepcopy(atmosphere_original)
                object = deepcopy(object_original)
            end
            heights[order2fit[l]] = height_trials[l][argmin(ϵ[l])]
            change_heights!(patches, atmosphere, object, observations, masks, heights, reconstruction=reconstruction)
        end
        println("Optimal Heights: $(heights)")
        opd_solve!(reconstruction, observations, atmosphere, object, masks, patches, verb="full", plot=false)
        object_solve!(reconstruction, observations, object, patches; verb="full", plot=false)
        object_original = deepcopy(object)
        atmosphere_original = deepcopy(atmosphere)
    end

    return ϵ, height_trials, atmosphere_original, object_original
end
