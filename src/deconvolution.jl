using FFTW
FFTW.set_num_threads(1)
using Statistics
using FourierTools
using LinearAlgebra
using OptimPackNextGen


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
    regularizers::Regularizers{T}
    ϕ_smoothing::Bool
    fwhm_schedule::Function
    function Reconstruction(
            observations;
            λ=[400.0],
            ndatasets=1,
            build_dim=256,
            niter_mfbd=20,
            indx_boot=[1:1],
            maxiter=10,
            gtol=(0.0, 1e-9),
            ϕ_smoothing=false,
            fwhm_schedule=x -> max(10 * (1 - x/niter_mfbd), 1.0),
            regularizers=[],
            verb=true,
            FTYPE=Float64
        )
        nλ = length(λ)
        Δλ = (nλ == 1) ? 1.0 : (maximum(λ) - minimum(λ)) / (nλ - 1)

        niter_boot = length(indx_boot)
        ϵ = zeros(FTYPE, niter_mfbd)
        if verb == true
            println("Setting up MFBD: $(nλ) wavelengths, $(niter_mfbd) iterations, $(niter_boot) aperture diverse channels")
        end

        if regularizers == []
            regularizers = Regularizers(FTYPE=FTYPE)
        end

        return new{FTYPE}(λ, nλ, Δλ, ndatasets, build_dim, niter_mfbd, indx_boot, niter_boot, maxiter, ϵ, gtol, regularizers, ϕ_smoothing, fwhm_schedule)
    end
end

mutable struct Helpers{T<:AbstractFloat}
    extractor::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    extractor_adj::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    # extractor_adj::Array{LazyAlgebra.Adjoint{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}, 4}
    refraction::Matrix{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}
    refraction_adj::Matrix{LazyAlgebra.Adjoint{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}}
    ift::Vector{Function}
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
    containers_pdim_real::Vector{Array{T, 3}}
    containers_pdim_cplx::Vector{Array{Complex{T}, 3}}
    @views function Helpers(atmosphere,
                     observations,
                     reconstruction,
                     object;
                     verb=true)
        if verb == true
            println("Setting up interpolators, buffers, fft/ifft plans, convolution/correlation plans")
        end
        FTYPE = gettype(reconstruction)

        ifft_threads = [setup_ifft(reconstruction.build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        extractor = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, reconstruction.ndatasets, observations[1].nepochs, atmosphere.nlayers, reconstruction.nλ)
        extractor_adj = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, reconstruction.ndatasets, observations[1].nepochs, atmosphere.nlayers, reconstruction.nλ)
        # extractor_adj = Array{LazyAlgebra.Adjoint{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}, 4}(undef, reconstruction.ndatasets, observations[1].nepochs, atmosphere.nlayers, reconstruction.nλ)
        refraction = Matrix{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}(undef, reconstruction.ndatasets, reconstruction.nλ)
        refraction_adj = Matrix{LazyAlgebra.Adjoint{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}}(undef, reconstruction.ndatasets, reconstruction.nλ)
        
        nthreads = Threads.nthreads()
        ϵ_threads = zeros(FTYPE, nthreads)
        g_threads_ϕ = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ, nthreads)
        g_threads_opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, nthreads)
        g_threads_obj = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, reconstruction.nλ, nthreads)
        ϕ_full = zeros(FTYPE, atmosphere.dim, atmosphere.dim, nthreads)
        ϕ_slice = Array{FTYPE, 3}(undef, reconstruction.build_dim, reconstruction.build_dim, nthreads)
        ϕ_composite = Array{FTYPE, 3}(undef, reconstruction.build_dim, reconstruction.build_dim, nthreads)
        P = Array{Complex{FTYPE}, 4}(undef, reconstruction.build_dim, reconstruction.build_dim, reconstruction.nλ, nthreads)
        p = Array{Complex{FTYPE}, 4}(undef, reconstruction.build_dim, reconstruction.build_dim, reconstruction.nλ, nthreads)
        Î_big = Array{FTYPE, 3}(undef, reconstruction.build_dim, reconstruction.build_dim, nthreads)
        c = Array{FTYPE, 3}(undef, reconstruction.build_dim, reconstruction.build_dim, nthreads)
        d = Array{Complex{FTYPE}, 3}(undef, reconstruction.build_dim, reconstruction.build_dim, nthreads)
        d2 = Array{FTYPE, 3}(undef, reconstruction.build_dim, reconstruction.build_dim, nthreads)
        containers_builddim_cplx = Array{Complex{FTYPE}, 3}(undef, reconstruction.build_dim, reconstruction.build_dim, nthreads)
        containers_builddim_real = Array{FTYPE, 3}(undef, reconstruction.build_dim, reconstruction.build_dim, nthreads)
        containers_sdim_real = Array{FTYPE, 3}(undef, atmosphere.dim, atmosphere.dim, nthreads)

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
            for w=1:reconstruction.nλ
                refraction[dd, w] = create_refraction_operator(reconstruction.λ[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale[w], reconstruction.build_dim, FTYPE=FTYPE)
                refraction_adj[dd, w] = refraction[dd, w]'
                for l=1:atmosphere.nlayers
                    for t=1:observations[dd].nepochs
                        extractor[dd, t, l, w] = create_extractor_operator(atmosphere.positions[:, t, l, w], atmosphere.dim, reconstruction.build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                        extractor_adj[dd, t, l, w] = create_extractor_adjoint(atmosphere.positions[:, t, l, w], atmosphere.dim, reconstruction.build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                    end
                end
            end
        end

        return new{FTYPE}(extractor, extractor_adj, refraction, refraction_adj, ifft_threads, ϕ_full, ϕ_slice, ϕ_composite, P, p, c, d, d2, Î_big, Î_small, r, ω, ϵ_threads, g_threads_ϕ, g_threads_obj, g_threads_opd, containers_builddim_real, containers_builddim_cplx, containers_sdim_real, containers_pdim_real, containers_pdim_cplx)
    end
end

@views function reconstruct_blind!(reconstruction, observations, atmosphere, object, masks, helpers, regularizers; verb="mfbd", plot=false)
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
    k_corr = Vector{typeof(plan_ccorr_psf_buffer(k, k)[2])}(undef, nthreads)
    o_conv = Matrix{typeof(plan_conv_psf_buffer(k, k)[2])}(undef, reconstruction.nλ, nthreads)
    o_corr = Matrix{typeof(plan_ccorr_psf_buffer(k, k)[2])}(undef, reconstruction.nλ, nthreads)
    for dd=1:reconstruction.ndatasets
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
            
            fwhm = reconstruction.fwhm_schedule((b-1)*reconstruction.niter_mfbd + current_iter)
            if (b > 1) || (fwhm <= 1)
                reconstruction.ϕ_smoothing = false
            end    

            k .= gaussian_kernel(reconstruction.build_dim, fwhm, FTYPE=FTYPE)
            for tid=1:nthreads
                k_conv[tid] = plan_conv_psf_buffer(k, k)[2]
                k_corr[tid] = plan_ccorr_psf_buffer(k, k)[2]
            end
            
            for w=1:reconstruction.nλ
                for tid=1:nthreads
                    o_conv[w, tid] = plan_conv_psf_buffer(k, object.object[:, :, w])[2]
                    o_corr[w, tid] = plan_ccorr_psf_buffer(k, object.object[:, :, w])[2]
                end
            end

            ## Reconstruct complex pupil
            if vm == true
                print("--> Reconstructing complex pupil ")
            end

            ## Reconstruct Phase
            crit_phase = (x, g) -> fg_phase(x, g, current_observations, atmosphere, current_masks, reconstruction, helpers, regularizers, o_conv, o_corr, k_conv, k_corr)
            vmlmb!(crit_phase, atmosphere.ϕ, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, fmin=0)
            # crit_opd = (x, g) -> @time fg_opd(x, g, current_observations, atmosphere, current_masks, reconstruction, helpers, regularizers, o_conv, o_corr, k_conv, k_corr)
            # vmlmb!(crit_opd, atmosphere.opd, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, fmin=0)
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

            crit_obj = (x, g) -> fg_object(x, g, current_observations, reconstruction, helpers, regularizers)
            vmlmb!(crit_obj, object.object, lower=0, verb=vo, mem=5, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, fmin=0)
            GC.gc()

            if plot == true
                o_obs[] = rotr90(dropdims(mean(object.object, dims=3), dims=3))
            end

            ## Compute final criterion
            reconstruction.ϵ[current_iter] = fg_object(object.object, similar(object.object), current_observations, reconstruction, helpers, regularizers)

            if vm == true
                print("--> ϵ:\t$(reconstruction.ϵ[current_iter])\n")
            end
        end
    end
end

@views function reconstruct_myopic!(reconstruction, observations, atmosphere, object, masks, helpers, regularizers; verb="mfbd", plot=false)
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
        ## Plots the object
        fig = Figure(size=(600, 600))
        o_ax = Axis(fig[1, 1])
        o_ax.title = "Object (mean)"
        hidedecorations!(o_ax)
        o_ax.aspect = DataAspect()
        global o_obs = Observable(rotr90(dropdims(mean(object.object, dims=3), dims=3)))
        o_plt = plot!(o_ax, o_obs, colormap=Reverse(:Greys))
        Colorbar(fig[1, 2], o_plt, width=10)
        trim!(fig.layout)
        display(fig)
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
            fill_psfs!(reconstruction, current_observations, atmosphere, object, current_masks, helpers)

            ## Reconstruct Object
            if vm == true
                print("--> object ")
            end

            crit_obj = (x, g) -> fg_object(x, g, current_observations, reconstruction, helpers, regularizers)
            vmlmb!(crit_obj, object.object, lower=0, mem=5, verb=vo, maxiter=reconstruction.maxiter, gtol=reconstruction.gtol, fmin=0)

            ## Compute final criterion
            reconstruction.ϵ[current_iter] = fg_object(object.object, helpers.g_threads_obj[:, :, :, 1], current_observations, reconstruction, helpers, regularizers)
        end
    end
end

@views function fill_psfs!(reconstruction, observations, atmosphere, object, masks, helpers)
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    iffts = [setup_ifft(reconstruction.build_dim, FTYPE=FTYPE) for tid=1:Threads.nthreads()]
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    scaleby_height = 1 .- atmosphere.heights ./ object.height
    ϕ_slices = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, Threads.nthreads())
    ϕ_composite = zeros(FTYPE, reconstruction.build_dim, reconstruction.build_dim, Threads.nthreads())
    for dd=1:ndatasets
        mask = masks[dd]
        observation = observations[dd]
        observation.psfs = zeros(FTYPE, observation.dim, observation.dim, observation.nsubaps, observation.nepochs, reconstruction.nλ)
        Threads.@threads :static for t=1:observation.nepochs
            tid = Threads.threadid()
            P = helpers.P[:, :, :, tid]
            p = helpers.p[:, :, :, tid]
            for n=1:observation.nsubaps
                for w=1:reconstruction.nλ
                    fill!(ϕ_composite[:, :, tid], zero(FTYPE))
                    for l=1:atmosphere.nlayers
                        extractor = create_extractor_operator(atmosphere.positions[:, t, l, w], atmosphere.dim, observation.dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                        position2phase!(ϕ_slices[:, :, tid], atmosphere.opd[:, :, l] .* (FTYPE(2pi)/reconstruction.λ[w]), extractor)
                        ϕ_composite[:, :, tid] .+= ϕ_slices[:, :, tid]
                    end
                    refraction = create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observation.ζ, observation.detector.pixscale[w], reconstruction.build_dim, FTYPE=FTYPE)
                    pupil2psf!(observation.psfs[:, :, n, t, w], helpers.containers_builddim_real[:, :, tid], mask.masks[:, :, n, w], P[:, :, w], p[:, :, w], atmosphere.A[:, :, t, w], ϕ_composite[:, :, tid], observation.α, mask.scale_psfs[w], iffts[tid], refraction)
                end
            end
        end
    end
end
