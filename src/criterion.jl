function loglikelihood_gaussian(data, model, ω)
    FTYPE = eltype(data)
    r = zeros(FTYPE, size(data))
    ϵ = FTYPE(loglikelihood_gaussian(r, data, model, ω))
    return ϵ
end

function loglikelihood_gaussian(r, data, model, ω)
    FTYPE = eltype(r)
    r .= model .- data
    ϵ = FTYPE(mapreduce(x -> x[1] * x[2]^2, +, zip(ω, r)))
    return ϵ
end

@views function fg_object(x, g, observations, reconstruction, patches)
    ## Optimized but unreadable
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    helpers = reconstruction.helpers
    patch_helpers = reconstruction.patch_helpers
    regularizers = reconstruction.regularizers
    fill!(g, zero(FTYPE))
    fill!(helpers.g_threads_obj, zero(FTYPE))
    fill!(helpers.ϵ_threads, zero(FTYPE))

    # o_obs[] = rotr90(x[:, :, 1])

    for dd=1:ndatasets
        observation = observations[dd]
        psfs = patches.psfs[dd]
        r = helpers.r[dd]
        ω = helpers.ω[dd]
        Î_small = helpers.Î_small[dd]
        fill!(observation.model_images, zero(FTYPE))
        Threads.@threads :static for t=1:observation.nepochs
            tid = Threads.threadid()
            for n=1:observation.nsubaps
                for np=1:patches.npatches
                    create_polychromatic_image!(observation.model_images[:, :, n, t], Î_small[:, :, tid], helpers.Î_big[:, :, tid], patches.w[:, :, np], helpers.containers_builddim_real[:, :, tid], x, psfs[:, :, np, n, t, :], reconstruction.λ, reconstruction.Δλ)
                end
                ω[:, :, tid] .= (observation.entropy[n, t] * observation.detector.rn^2)^(-1)
                helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid])
                gradient_object!(helpers.g_threads_obj[:, :, :, tid], r[:, :, tid], ω[:, :, tid], helpers.Î_big[:, :, tid], psfs[:, :, :, n, t, :], patches.w, patches.npatches, reconstruction.Δλ, reconstruction.nλ)
            end
        end
    end

    ϵ = FTYPE(sum(helpers.ϵ_threads))
    for tid=1:Threads.nthreads()
        g .+= helpers.g_threads_obj[:, :, :, tid]
    end

    for w=1:reconstruction.nλ
        ϵ += regularizers.o_reg(x[:, :, w], g[:, :, w], regularizers.βo)
    end
    ϵ += regularizers.λ_reg(x, g, regularizers.βλ)

    return ϵ
end

@views function gradient_object!(g, r, ω, image_big, psfs, patch_weights, npatches, Δλ, nλ)
    r .*= ω
    block_replicate!(image_big, r)
    for np=1:npatches
        for w=1:nλ
            g[:, :, w] .+= (2*Δλ) .* patch_weights[:, :, np] .* ccorr_psf(image_big, psfs[:, :, np, w])
        end
    end
end

# @views function gradient_object!(g, r, ω, image_big, psfs, entropy, nλ)
#     r .= 2 .* ω .* r .- (ω .* r).^2 ./ entropy
#     block_replicate!(image_big, r)
#     for w=1:nλ
#         g[:, :, w] .+= 2 .* ccorr_psf(image_big, psfs[:, :, w])
#     end
# end

@views function fg_opd(x, g, observations, atmosphere, masks, patches, reconstruction, o_conv, o_corr, k_conv, k_corr)
    ## Optimized but unreadable
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    helpers = reconstruction.helpers
    patch_helpers = reconstruction.patch_helpers
    fill!(g, zero(FTYPE))
    fill!(helpers.g_threads_opd, zero(FTYPE))
    fill!(helpers.ϵ_threads, zero(FTYPE))

    for dd=1:ndatasets
        observation = observations[dd]
        psfs = patches.psfs[dd]
        mask = masks[dd]
        scale_psf = mask.scale_psfs
        refraction = helpers.refraction[dd, :]
        refraction_adj = helpers.refraction_adj[dd, :]
        r = helpers.r[dd]
        ω = helpers.ω[dd]
        α = observation.α
        Î_small = helpers.Î_small[dd]
        fill!(observation.model_images, zero(FTYPE))     
        Threads.@threads :static for t=1:observation.nepochs
            tid = Threads.threadid()
            extractor = patch_helpers.extractor[t, :, :, :]
            extractor_adj = patch_helpers.extractor_adj[t, :, :, :]
            Î_big = helpers.Î_big[:, :, tid]
            ϕ_slices = patches.ϕ_slices[:, :, :, t, :, :]
            ϕ_composite = patches.ϕ_composite[:, :, :, t, :]
            for n=1:observation.nsubaps
                P = patch_helpers.P[:, :, :, :, tid]
                p = patch_helpers.p[:, :, :, :, tid]
                for np=1:patches.npatches
                    for w=1:reconstruction.nλ
                        fill!(ϕ_composite[:, :, np, w], zero(FTYPE))
                        for l=1:atmosphere.nlayers
                            ## Aliases don't allocate
                            helpers.containers_sdim_real[:, :, tid] .= FTYPE(2pi) ./ reconstruction.λ[w] .* x[:, :, l]
                            position2phase!(ϕ_slices[:, :, np, l, w], helpers.containers_sdim_real[:, :, tid], extractor[np, l, w])
                            ϕ_composite[:, :, np, w] .+= ϕ_slices[:, :, np, l, w]
                        end

                        if reconstruction.smoothing == true
                            ϕ_composite[:, :, np, w] .= k_conv[tid](ϕ_composite[:, :, np, w])
                        end

                        pupil2psf!(psfs[:, :, np, n, t, w], helpers.containers_builddim_real[:, :, tid], mask.masks[:, :, n, w], P[:, :, np, w], p[:, :, np, w], observation.A[:, :, n, np, t, w], ϕ_composite[:, :, np, w], α, scale_psf[w], helpers.ift[tid], refraction[w])
                    end
                    create_polychromatic_image!(observation.model_images[:, :, n, t], Î_small[:, :, tid], Î_big, o_conv[:, np, tid], psfs[:, :, np, n, t, :], reconstruction.λ, reconstruction.Δλ)
                end
                ω[:, :, tid] .= (observation.entropy[n, t] * observation.detector.rn^2)^(-1)
                helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid])
                gradient_opd!(helpers.g_threads_opd[:, :, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λ, reconstruction.Δλ, reconstruction.nλ, α, atmosphere.nlayers, o_corr[:, :, tid], patches.npatches, reconstruction.smoothing, k_corr[tid], refraction_adj, extractor_adj, helpers.ift[tid], helpers.containers_builddim_real[:, :, tid], helpers.containers_sdim_real[:, :, tid])
            end
        end
    end

    ϵ = sum(helpers.ϵ_threads)
    for tid=1:Threads.nthreads()
        g .+= helpers.g_threads_opd[:, :, :, tid]
    end

    # Apply regularization
    # for w=1:reconstruction.nλ
    #     for l=1:atmosphere.nlayers
    #         ϵ += reconstruction.regularizers.ϕ_reg(x[:, :, l, w], g[:, :, l, w], regularizers.βϕ, mask=atmosphere.mask[:, :, l, w])
    #     end
    # end

    return ϵ
end

@views function gradient_opd!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, α, nlayers, o_corr, npatches, smoothing, k_corr, refraction_adj, extractor_adj, ifft_prealloc, container_builddim_real, container_sdim_real)
    FTYPE = eltype(r)
    r .*= ω
    block_replicate!(c, r)
    conj!(p)
    for np=1:npatches
        for w=1:nλ
            d2 .= o_corr[w, np](c)  # <--
            container_builddim_real .= d2
            mul!(d2, refraction_adj[w], container_builddim_real)

            p[:, :, np, w] .*= d2
            ifft_prealloc(d, p[:, :, np, w])
            d .*= P[:, :, np, w]
            d2 .= imag.(d)
            d2 .*= FTYPE(-8pi) * Δλ/λ[w] * α

            if smoothing == true
                d2 .= k_corr(d2)
            end

            for l=1:nlayers
                mul!(container_sdim_real, extractor_adj[np, l, w], d2)
                g[:, :, l] .+= container_sdim_real
            end
        end
    end
end
