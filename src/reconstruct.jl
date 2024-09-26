using FFTW
using Statistics
using FourierTools
using LinearAlgebra
using OptimPackNextGen


if Threads.nthreads() == 1
    FFTW.set_num_threads(4)
else
    FFTW.set_num_threads(1)
end

function ConstantSchedule()
    function constant(x)
        return x
    end

    return constant
end

function ReciprocalDecaySchedule(maxval, niters, minval)
    function reciprocal(x)
        return max(maxval * (1 - (x-1)/niters), minval)
    end

    return reciprocal
end

mutable struct Helpers{T<:AbstractFloat}
    extractor::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    extractor_adj::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    refraction::Matrix{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}
    refraction_adj::Matrix{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}
    ift::Vector{Function}
    convolve::Vector{Function}
    correlate::Vector{Function}
    autocorr::Vector{Function}
    ϕ_full::Array{T, 3}
    ϕ_slice::Array{T, 3}
    ϕ_composite::Array{T, 3}
    o_conv::Array{Function, 3}
    o_corr::Array{Function, 3}
    smoothing_kernel::Matrix{T}
    k_conv::Vector{Function}
    k_corr::Vector{Function}
    P::Array{Complex{T}, 4}
    p::Array{Complex{T}, 4}
    c::Array{T, 3}
    d::Array{Complex{T}, 3}
    d2::Array{T, 3}
    Î_big::Array{T, 3}
    Î_small::Vector{Array{T, 3}}
    r::Vector{Array{T, 3}}
    ω::Vector{Array{T, 3}}
    M::Vector{Matrix{T}}
    ϵ_threads::Vector{T}
    g_threads_obj::Array{T, 4}
    g_threads_opd::Array{T, 4}
    containers_builddim_real::Array{T, 3}
    containers_builddim_cplx::Array{Complex{T}, 3}
    containers_sdim_real::Array{T, 3}
    containers_sdim_cplx::Array{T, 3}
    containers_pdim_real::Vector{Array{T, 3}}
    containers_pdim_cplx::Vector{Array{Complex{T}, 3}}
    function Helpers(atmosphere,
                     observations,
                     object,
                     patches;
                     λtotal=atmosphere.λ,
                     build_dim=size(object.object, 1),
                     ndatasets=length(observations),
                     verb=true,
                     FTYPE=gettype(atmosphere))
        if verb == true
            println(" |-> Creating fft/ifft plans and convolution/correlation plans")
        end
        nλtotal = length(λtotal)
        ifft_threads = [setup_ifft(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        conv_threads = [setup_conv(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        corr_threads = [setup_corr(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        autocorr_threads = [setup_autocorr(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]

        nthreads = Threads.nthreads()
        if verb == true
            println(" |-> Creating computation buffers for $(nthreads) threads")
        end
        ϵ_threads = zeros(FTYPE, nthreads)
        g_threads_opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, nthreads)
        g_threads_obj = zeros(FTYPE, build_dim, build_dim, object.nλ, nthreads)
        ϕ_full = zeros(FTYPE, atmosphere.dim, atmosphere.dim, nthreads)
        ϕ_slice = Array{FTYPE, 3}(undef, build_dim, build_dim, nthreads)
        ϕ_composite = zeros(FTYPE, build_dim, build_dim, nthreads)
        o_conv = Array{Function, 3}(undef, object.nλ, patches.npatches, nthreads)
        o_corr = Array{Function, 3}(undef, object.nλ, patches.npatches, nthreads)
        smoothing_kernel = zeros(FTYPE, build_dim, build_dim)
        k_conv = Vector{Function}(undef, nthreads)
        k_corr = Vector{Function}(undef, nthreads)
        P = zeros(Complex{FTYPE}, build_dim, build_dim, nλtotal, nthreads)
        p = zeros(Complex{FTYPE}, build_dim, build_dim, nλtotal, nthreads)
        Î_big = zeros(FTYPE, build_dim, build_dim, nthreads)
        c = zeros(FTYPE, build_dim, build_dim, nthreads)
        d = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
        d2 = zeros(FTYPE, build_dim, build_dim, nthreads)
        containers_builddim_cplx = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
        containers_builddim_real = zeros(FTYPE, build_dim, build_dim, nthreads)
        containers_sdim_real = zeros(FTYPE, atmosphere.dim, atmosphere.dim, nthreads)
        containers_sdim_cplx = zeros(Complex{FTYPE}, atmosphere.dim, atmosphere.dim, nthreads)
        ndatasets = length(observations)
        r = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        ω = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        M = Vector{Matrix{FTYPE}}(undef, ndatasets)
        Î_small = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        containers_pdim_real = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        containers_pdim_cplx = Vector{Array{Complex{FTYPE}, 3}}(undef, ndatasets)

        if verb == true
            println(" |-> Creating punch-out and refraction interpolators/adjoints")
        end
        extractor = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, ndatasets, observations[1].nepochs, atmosphere.nlayers, nλtotal)
        extractor_adj = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, ndatasets, observations[1].nepochs, atmosphere.nlayers, nλtotal)
        refraction = Matrix{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}(undef, ndatasets, nλtotal)
        refraction_adj = Matrix{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}(undef, ndatasets, nλtotal)
        for dd=1:ndatasets
            r[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
            ω[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
            M[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim)
            M[dd][observations[dd].dim÷2+1, observations[dd].dim÷2+1] = 1
            Î_small[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
            containers_pdim_real[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
            containers_pdim_cplx[dd] = zeros(Complex{FTYPE}, observations[dd].dim, observations[dd].dim, nthreads)
            scaleby_wavelength = [observations[dd].detector.λ_nyquist / λtotal[w] for w=1:nλtotal]
            # scaleby_height = 1 .- atmosphere.heights ./ object.height
            scaleby_height = layer_scale_factors(atmosphere.heights, object.height)
            # Dmeta = observations[dd].D .+ (object.fov/206265) .* (atmosphere.heights .* 1000)
            # scaleby_height = Dmeta ./ observations[dd].D
            Threads.@threads for w=1:nλtotal
                refraction[dd, w] = create_refraction_operator(λtotal[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale, build_dim, FTYPE=FTYPE)
                refraction_adj[dd, w] = create_refraction_adjoint(λtotal[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale, build_dim, FTYPE=FTYPE)
                for l=1:atmosphere.nlayers
                    for t=1:observations[dd].nepochs
                        extractor[dd, t, l, w] = create_extractor_operator(atmosphere.positions[:, t, l, w], atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                        extractor_adj[dd, t, l, w] = create_extractor_adjoint(atmosphere.positions[:, t, l, w], atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                    end
                end
            end
        end

        return new{FTYPE}(extractor, extractor_adj, refraction, refraction_adj, ifft_threads, conv_threads, corr_threads, autocorr_threads, ϕ_full, ϕ_slice, ϕ_composite, o_conv, o_corr, smoothing_kernel, k_conv, k_corr, P, p, c, d, d2, Î_big, Î_small, r, ω, M, ϵ_threads, g_threads_obj, g_threads_opd, containers_builddim_real, containers_builddim_cplx, containers_sdim_real, containers_sdim_cplx, containers_pdim_real, containers_pdim_cplx)
    end
end

mutable struct PatchHelpers{T<:AbstractFloat}
    extractor::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    extractor_adj::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    P::Array{Complex{T}, 5}
    p::Array{Complex{T}, 5}
    function PatchHelpers(patches, observations, atmosphere, object; λtotal=atmosphere.λ, build_dim=size(object.object, 1), verb=true, FTYPE=gettype(atmosphere))
        if verb == true
            println(" |-> Creating patch extractors and buffers")
        end
        nλtotal = length(λtotal)
        P = zeros(Complex{FTYPE}, build_dim, build_dim, patches.npatches, nλtotal, Threads.nthreads())
        p = zeros(Complex{FTYPE}, build_dim, build_dim, patches.npatches, nλtotal, Threads.nthreads())
        
        scaleby_wavelength = atmosphere.λ_nyquist ./ λtotal
        scaleby_height = layer_scale_factors(atmosphere.heights, object.height)
        extractor = create_patch_extractors(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
        extractor_adj = create_patch_extractors_adjoint(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
        return new{FTYPE}(extractor, extractor_adj, P, p)
    end
end

mutable struct Reconstruction{T<:AbstractFloat}
    λ::Vector{T}
    λtotal::Vector{T}
    nλ::Int64
    nλint::Int64
    nλtotal::Int64
    Δλ::T
    Δλtotal::T
    ndatasets::Int64
    build_dim::Int64
    weight_function::Function
    gradient_object::Function
    gradient_opd::Function
    niter_mfbd::Int64
    indx_boot::Vector{UnitRange{Int64}}
    niter_boot::Int64
    maxiter::Int64
    ϵ::T
    grtol::T
    frtol::T
    xrtol::T
    maxeval::Dict{String, Int64}
    regularizers::Regularizers{T}
    smoothing::Bool
    fwhm_schedule::Function
    helpers::Helpers{T}
    patch_helpers::PatchHelpers{T}
    verb_levels::Dict{String, Bool}
    plot::Bool
    figures::ReconstructionFigures
    function Reconstruction(
            atmosphere,
            observations,
            object,
            patches;
            λmin=400.0,
            λmax=1000.0,
            nλ=1,
            nλint=1,
            ndatasets=length(observations),
            build_dim=size(object.object, 1),
            weight_function=gaussian_weighting,
            gradient_object=gradient_object_gaussiannoise!,
            gradient_opd=gradient_opd_gaussiannoise!,
            niter_mfbd=10,
            indx_boot=[1:dd for dd=1:ndatasets],
            maxiter=10,
            maxeval=Dict("object"=>100000, "opd"=>100000),
            grtol=1e-9,
            frtol=1e-9,
            xrtol=1e-9,
            smoothing=false,
            maxFWHM=10.0,
            minFWHM=0.1,
            fwhm_schedule=ReciprocalDecaySchedule(maxFWHM, niter_mfbd, minFWHM),
            regularizers=[],
            helpers=[],
            patch_helpers=[],
            verb=true,
            mfbd_verb_level="full",
            plot=true,
            FTYPE = gettype(atmosphere)
        )
        nλtotal = nλ * nλint
        λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
        λtotal = (nλtotal == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλtotal))
        Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
        Δλtotal = (nλtotal == 1) ? 1.0 : (λmax - λmin) / (nλtotal - 1)
        niter_boot = length(indx_boot)
        ϵ = zero(FTYPE)
        if verb == true
            println("Setting up MFBD: $(nλ) wavelengths ($(nλint) integrated planes), $(niter_mfbd) iterations, $(ndatasets) data channels")
            println(" |-> Noise weighting: $(weight_function), object gradient: $(gradient_object), opd gradient: $(gradient_opd)")
            println(" |-> Max iterations: $(maxiter), Max evaluations (OPD): $(maxeval["opd"]), (object): $(maxeval["object"])")
            println(" |-> Smoothing: $(smoothing) (schedule: $(fwhm_schedule), Max FWHM: $(maxFWHM), Min FWHM: $(minFWHM))")
        end

        if helpers == []
            helpers = Helpers(
                atmosphere, 
                observations,
                object,
                patches,
                λtotal=λtotal,
                verb=verb
            );
        end

        if patch_helpers == []
            patch_helpers = PatchHelpers(
                patches,
                observations,
                atmosphere,
                object,
                λtotal=λtotal,
                verb=verb
            )
        end

        if regularizers == []
            regularizers = Regularizers(verb=verb, FTYPE=FTYPE)
        end

        patches.A = Vector{Array{FTYPE, 6}}(undef, ndatasets)
        patches.ϕ_slices = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[1].nepochs, atmosphere.nlayers, nλtotal)
        patches.ϕ_composite = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[1].nepochs, nλtotal)
        patches.psfs = Vector{Array{FTYPE, 6}}(undef, ndatasets)
        patches.broadband_psfs = Vector{Array{FTYPE, 5}}(undef, ndatasets)
        for dd=1:ndatasets
            patches.A[dd] = ones(FTYPE, build_dim, build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, nλtotal)
            patches.psfs[dd] = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, nλ)
            patches.broadband_psfs[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs)
            observations[dd].psfs = zeros(FTYPE, build_dim, build_dim, observations[dd].nsubaps, observations[dd].nepochs, nλ)
            observations[dd].model_images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
            observations[dd].w = findall((observations[dd].detector.filter.response .* observations[dd].detector.qe) .> 0)
        end

        if mfbd_verb_level == "full"
            vm = true
            vo = true
        elseif mfbd_verb_level == "mfbd"
            vm = true
            vo = false
        elseif mfbd_verb_level == "silent"
            vm = false
            vo = false
        end
        verb_levels = Dict("vm"=>vm, "vo"=>vo)

        figs = ReconstructionFigures()
        figs.object_fig, figs.object_ax, figs.object_obs = plot_object(object, show=false)
        figs.opd_fig, figs.opd_ax, figs.opd_obs = plot_layers(atmosphere, show=false)
        if plot == true
            display(GLMakie.Screen(), figs.object_fig)
            display(GLMakie.Screen(), figs.opd_fig)
        end

        if verb == true
            println("")
        end

        return new{FTYPE}(λ, λtotal, nλ, nλint, nλtotal, Δλ, Δλtotal, ndatasets, build_dim, weight_function, gradient_object, gradient_opd, niter_mfbd, indx_boot, niter_boot, maxiter, ϵ, grtol, frtol, xrtol, maxeval, regularizers, smoothing, fwhm_schedule, helpers, patch_helpers, verb_levels, plot, figs)
    end
end

function gaussian_weighting(entropy, image, rn)
    return 1 / (entropy * rn^2)
end

function mixed_weighting(entropy, image, rn)
    return 1 ./ (entropy .* (image .+ rn^2))
end


@views function object_solve!(reconstruction, observations, object, patches)
    FTYPE = gettype(reconstruction)
    reconstruction.ϵ = zero(FTYPE)  
    crit_obj = (x, g) -> fg_object(x, g, observations, reconstruction, patches)
    vmlmb!(crit_obj, object.object, lower=0, fmin=0, verb=reconstruction.verb_levels["vo"], maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["object"])
end

@views function opd_solve!(reconstruction, observations, atmosphere, object, masks, patches)
    FTYPE = gettype(reconstruction)
    reconstruction.ϵ = zero(FTYPE)
    helpers = reconstruction.helpers
    fill_object_subplanes!(helpers.object_full, object.object, reconstruction.nλ, reconstruction.nλobject, reconstruction.Δλ, reconstruction.Δλobject, reconstruction.nsubλ)    
    absolute_iter = (b-1)*reconstruction.niter_boot + current_iter
    update_hyperparams(reconstruction, absolute_iter)
    preconvolve_smoothing(reconstruction)
    preconvolve_object(reconstruction, patches, object)

    crit_opd = (x, g) -> fg_opd(x, g, observations, atmosphere, masks, patches, reconstruction)
    vmlmb!(crit_opd, atmosphere.opd, verb=reconstruction.verb_levels["vo"], mem=5, fmin=0, maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["opd"])
end

function reconstruct_blind!(reconstruction, observations, atmosphere, object, masks, patches; closeplots=true, write=false, folder="", id="")
    FTYPE = gettype(reconstruction)
    for b=1:reconstruction.niter_boot
        current_observations = observations[reconstruction.indx_boot[b]]
        current_masks = masks[reconstruction.indx_boot[b]]
        reconstruction.ϵ = zero(FTYPE)
        for current_iter=1:reconstruction.niter_mfbd
            if reconstruction.verb_levels["vm"] == true
                print("Bootstrap Iter: $(b) MFBD Iter: $(current_iter) ")
            end
        
            absolute_iter = (b-1)*reconstruction.niter_boot + current_iter
            update_hyperparams(reconstruction, absolute_iter)
            preconvolve_smoothing(reconstruction)
            preconvolve_object(reconstruction, patches, object)

            ## Reconstruct complex pupil
            if reconstruction.verb_levels["vm"] == true
                print("--> Reconstructing complex pupil ")
            end

            ## Reconstruct Phase
            crit_opd = (x, g) -> fg_opd(x, g, current_observations, atmosphere, current_masks, patches, reconstruction)
            vmlmb!(crit_opd, atmosphere.opd, verb=reconstruction.verb_levels["vo"], fmin=0, mem=5, maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["opd"])
            update_layer_figure(atmosphere.opd, atmosphere, reconstruction)

            ## Reconstruct Object
            if reconstruction.verb_levels["vm"] == true
                print("--> object ")
            end

            crit_obj = (x, g) -> fg_object(x, g, current_observations, reconstruction, patches)
            vmlmb!(crit_obj, object.object, lower=0, fmin=0, verb=reconstruction.verb_levels["vo"], maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["object"])
            update_object_figure(dropdims(mean(object.object, dims=3), dims=3), reconstruction)

            ## Compute final criterion
            if reconstruction.verb_levels["vm"] == true
                print("--> ϵ:\t$(reconstruction.ϵ)\n")
            end

            if write == true
                writefits(observations[1].model_images, "$(folder)/models_ISH1x1_recon$(id).fits")
                writefits(observations[2].model_images, "$(folder)/models_ISH6x6_recon$(id).fits")
                # writefits(patches.psfs[1], "$(folder)/psfs_ISH1x1_recon$(id).fits")
                writefits(object.object, "$(folder)/object_recon$(id).fits")
                writefits(atmosphere.opd, "$(folder)/opd_recon$(id).fits")
                writefile([reconstruction.ϵ], "$(folder)/recon$(id).dat")
            end
            GC.gc()
        end
        # reconstruction.smoothing = false
    end

    if closeplots == true
        GLMakie.closeall()
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
    height_trials = reverse([collect(FTYPE, hmin[l]:hstep[l]:hmax[l]) for l=1:nlayers2fit])
    # [heights[order2fit[l]] = height_trials[l][1] for l=1:nlayers2fit]
    ϵ = [zeros(FTYPE, length(height_trials[l])) for l=1:nlayers2fit]
    [fill!(ϵ[l], FTYPE(NaN)) for l=1:nlayers2fit]
    atmosphere_original = deepcopy(atmosphere)
    object_original = deepcopy(object)

    heights_fig, heights_ax, heights_obs, ϵ_obs = plot_heights(atmosphere, heights=height_trials, ϵ=ϵ, show=true)
    figs = reconstruction.figures
    figs.heights_fig = heights_fig
    figs.heights_ax = heights_ax
    figs.heights_obs = heights_obs
    figs.ϵ_obs = ϵ_obs

    for it=1:niters
        atmosphere = deepcopy(atmosphere_original)
        [fill!(ϵ[l], FTYPE(NaN)) for l=1:nlayers2fit]
        for l=1:nlayers2fit
            for h=1:length(height_trials[l])
                heights[order2fit[l]] = height_trials[l][h]
                print("\tHeight: $(heights)\t")
                change_heights!(patches, atmosphere, object, observations[end], masks[end], heights, reconstruction=reconstruction, verb=false)
                reconstruct_blind!(reconstruction, observations, atmosphere, object, masks, patches, closeplots=false)
                ϵ[l][h] = sum(reconstruction.ϵ[1])
                println("ϵ: $(ϵ[l][h])")
                atmosphere = deepcopy(atmosphere_original)
                object = deepcopy(object_original)
                update_heights_figure(height_trials, ϵ, atmosphere, reconstruction)
            end
            heights[order2fit[l]] = height_trials[l][argmin(ϵ[l])]
            change_heights!(patches, atmosphere, object, observations[end], masks[end], heights, reconstruction=reconstruction, verb=false)
        end
        println("Optimal Heights: $(heights)")
        reconstruct_blind!(reconstruction, observations, atmosphere, object, masks, patches, closeplots=false)
        object_original = deepcopy(object)
        atmosphere_original = deepcopy(atmosphere)
        update_heights_figure(height_trials, ϵ, atmosphere, reconstruction)
    end
    GLMakie.closeall()

    return ϵ, height_trials, atmosphere, object
end

@views function preconvolve_object(reconstruction, patches, object)
    helpers = reconstruction.helpers
    for w=1:object.nλ
        for np=1:patches.npatches
            for tid=1:Threads.nthreads()
                helpers.o_conv[w, np, tid] = plan_conv_psf_buffer(object.object[:, :, w], patches.w[:, :, np] .* object.object[:, :, w])[2]
                helpers.o_corr[w, np, tid] = plan_ccorr_psf_buffer(object.object[:, :, w], patches.w[:, :, np] .* object.object[:, :, w])[2]
            end
        end
    end
end

@views function preconvolve_smoothing(reconstruction)
    helpers = reconstruction.helpers
    for tid=1:Threads.nthreads()
        helpers.k_conv[tid] = plan_conv_psf_buffer(helpers.smoothing_kernel, helpers.smoothing_kernel)[2]
        helpers.k_corr[tid] = plan_ccorr_psf_buffer(helpers.smoothing_kernel, helpers.smoothing_kernel)[2]
    end
end

function update_hyperparams(reconstruction, iter)
    FTYPE = gettype(reconstruction)
    regularizers = reconstruction.regularizers
    helpers = reconstruction.helpers
    regularizers.βo *= regularizers.βo_schedule(iter)
    regularizers.βopd *= regularizers.βopd_schedule(iter)
    fwhm = reconstruction.fwhm_schedule(iter)
    helpers.smoothing_kernel .= gaussian_kernel(reconstruction.build_dim, fwhm, FTYPE=FTYPE)
end