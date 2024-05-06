using JLD2
using FITSIO
using Statistics
using FourierTools
using DelimitedFiles
import Interpolations: LinearInterpolation, Line


function gettype(T)
    return typeof(T).parameters[1]
end

function writefits(x, filename; verb=true, header=nothing)
    f = FITS(filename, "w")
    if header !== nothing
        header = FITSHeader(header...)
    end
    write(f, x, header=header)
    close(f)
    if verb == true
        print("Written to "); printstyled("$(filename)\n", color=:red)
    end
end

function writefile(x::Vector{<:AbstractFloat}, filename; header="", verb=true)
    f = open(filename, "w")
    if header != ""
        write(f, header * "\n")
    end
    for i=1:length(x)
        write(f, "$(i)\t$(x[i])\n")
    end
    close(f)
    if verb == true
        print("Written to "); printstyled("$(filename)\n", color=:red)
    end
end

function writefile(x, y, filename; header="", verb=true)
    f = open(filename, "w")
    if header != ""
        write(f, header * "\n")
    end
    for i=1:length(x)
        write(f, "$(x[i])\t$(y[i])\n")
    end
    close(f)
    if verb == true
        print("Written to "); printstyled("$(filename)\n", color=:red)
    end
end

function writeobject(x, filename; verb=true)
    jldsave(filename; x)
    if verb == true
        print("Written to "); printstyled("$(filename)\n", color=:red)
    end
end

function readobject(filename::String)
    return read(jldopen(filename, "r"), "x")
end

function readfile(filename)
    data = readdlm(filename)
    return data[:, 1], data[:, 2]
end

function readqe(filename; λ=[])
    λraw, qe_raw = readfile(filename)
    if λ != []
        itp = LinearInterpolation(λraw, qe_raw, extrapolation_bc=Line())
        qe = itp(λ)
    else
        λ, qe = λraw, qe_raw
    end

    return λ, qe
end

@views function readimages(file::String; FTYPE=Float64)
    images = readfits(file, FTYPE=FTYPE)
    nsubaps = size(images, 3);
    nepochs = size(images, 4);
    dim = size(images, 1);
    return images, nsubaps, nepochs, dim
end

function calculate_entropy(x)
    p = x ./ sum(x);
    ix = findall(x .> 0);
    return -sum(p[ix] .* log2.(p[ix]));
end

function readmasks(file::String; FTYPE=Float64)
    masks = readfits(file, FTYPE=FTYPE)
    hdu = FITS(file)[1]
    λstart = read_key(hdu, "WAVELENGTH_START")[1]
    λend = read_key(hdu, "WAVELENGTH_END")[1]
    nλ = read_key(hdu, "WAVELENGTH_STEPS")[1]
    λ = FTYPE.(collect(range(λstart, stop=λend, length=nλ)))
    return masks, λ
end

function readmasks(files::Vector{String}; FTYPE=Float64)
    nfiles = length(files)
    masks = Array{Array{FTYPE}}(undef, nfiles)
    for i=1:nfiles
        masks[i] = readfits(files[i], FTYPE=FTYPE)
    end

    hdu = FITS(files[1])[1]
    λstart = read_key(hdu, "WAVELENGTH_START")[1]
    λend = read_key(hdu, "WAVELENGTH_END")[1]
    nλ = read_key(hdu, "WAVELENGTH_STEPS")[1]
    λ = FTYPE.(collect(range(λstart, stop=λend, length=nλ)))

    masks = cat(masks..., dims=3)
    return masks, λ
end

function readfits(file; FTYPE=Float64)
    return FTYPE.(read(FITS(file)[1]))
end

function gaussian_kernel(dim, fwhm; FTYPE=Float64)
    coord = (1:dim) .- (dim÷2 + 1)
    rr = hypot.(coord, coord')
    σ = fwhm / 2.35482
    k = exp.(-rr.^2 ./ σ^2)
    k ./= sum(k)
    return FTYPE.(k)
end

@views function shift_and_add(images, nsubaps, nepochs, pdim; FTYPE=Float64)
    image_sum = zeros(FTYPE, pdim, pdim)
    kernel = zeros(FTYPE, pdim, pdim)
    for n=1:nsubaps
        for t=1:nepochs
            fill!(kernel, zero(FTYPE))
            center = Tuple(argmax(images[:, :, n, t]))
            Δy, Δx = (pdim÷2)+1 - center[1], (pdim÷2)+1 - center[2]
            shift = CartesianIndex((pdim÷2+1 + Δy, pdim÷2+1 + Δx))
            kernel[shift] = 1.0
            image_sum .+= conv_psf(images[:, :, n, t], kernel)
        end
    end

    return FTYPE.(image_sum ./ (nsubaps*nepochs))
end

function block_reduce(image, newdim)
    FTYPE = eltype(image)
    dim = size(image, 1)
    if newdim != dim
        newimage = zeros(FTYPE, newdim, newdim)
        block_reduce!(newimage, image)
    else
        newimage = image
    end
    
    return newimage
end

@views function block_reduce!(newimage, image)
    dim = size(image, 1)
    newdim = size(newimage, 1)
    pixperbin = dim ÷ newdim
    for i=1:newdim
        for j=1:newdim
            newimage[i, j] = sum(image[(i-1)*pixperbin + 1:i*pixperbin, (j-1)*pixperbin + 1:j*pixperbin])
        end
    end
end

function block_replicate(image, newdim)
    FTYPE = eltype(image)
    dim = size(image, 1)
    if newdim != dim
        newimage = zeros(FTYPE, newdim, newdim)
        block_replicate!(newimage, image)
    else
        newimage = image
    end
    return newimage
end

@views function block_replicate!(newimage, image)
    dim = size(image, 1)
    newdim = size(newimage, 1)
    pixperbin = newdim ÷ dim
    for i=1:dim
        for j=1:dim
            newimage[(i-1)*pixperbin + 1:i*pixperbin, (j-1)*pixperbin + 1:j*pixperbin] .= image[i, j] / pixperbin^2
        end
    end
end

function fit_plane(ϕ, mask)
    dim = size(ϕ, 1)
    plane = zeros(dim, dim)
    ix = (mask .> 0)
    X = (collect(1:dim)' .* ones(dim))
    Y = X'
    Z = ϕ[ix]
    N = length(X[ix])
    M = [X[ix] Y[ix] ones(N)]

    fit = inv(M' * M) * M' * Z
    ll = sqrt( fit[1]^2 + fit[2]^2 + 1 )
    a, b, c, d = -fit[1]/ll, -fit[2] / ll, 1 / ll, -fit[3] / ll

    plane[ix] = -((a/c .* X[ix]) .+ (b/c .* Y[ix]) .+ (d/c))
    return plane
end

function crop(x, newdim)
    dim = size(x, 1)
    if dim == newdim
        return x
    else
        ix1 = dim÷2-newdim÷2
        ix2 = dim÷2+newdim÷2-1
        return x[ix1:ix2, ix1:ix2]
    end
end

function smooth_to_rmse!(ϕ_smooth, ϕ, rms_target, mask, dim; FTYPE=Float64)
    rms = Inf
    fwhm = dim/10
    while (rms-rms_target > 1e-3)
        k = gaussian_kernel(dim, fwhm, FTYPE=FTYPE)
        ϕ_smooth .= conv_psf(ϕ, k)
        rms = sqrt(mean((ϕ_smooth[mask .> 0] .- ϕ[mask .> 0]).^2))
        fwhm -= fwhm*0.01
    end
end

function bartlett_hann1d(n, N)
    return 0.62 .- 0.48 .* abs.(n ./ N .- 0.5) .- 0.38 .* cos.(2pi .* n ./ N)
end

function bartlett_hann1d_centered(n, N)
    nn = n .+ (N+1 ÷ 2)
    return bartlett_hann1d(nn, N)
end

function bartlett_hann2d(i, j, N)
    return bartlett_hann1d(i, N) .* bartlett_hann1d(j, N)
end

function super_gaussian(dim, σ, n; FTYPE=Float64)
    x = (1:dim) .- (dim÷2+1)
    r = hypot.(x, x')
    sg = exp.(-(r ./ σ).^n)
    return FTYPE.(sg)
end

function vega_spectrum(; λ=[])
    file = "data/alpha_lyr_stis_011.fits"
    λ, flux = read_spectrum(file, λ=λ)
    return λ, flux
end

function solar_spectrum(; λ=[])
    file = "data/sun_reference_stis_002.fits"
    λ, flux = read_spectrum(file, λ=λ)
    return λ, flux
end

function read_spectrum(file; λ=[])
    h = 6.626196e-27  # erg⋅s
    c = 2.997924562e17  # nm/s

    λ₀ = read(FITS(file)[2], "WAVELENGTH")  # Å
    λ₀ ./= 10  # convert to nm
    flux = read(FITS(file)[2], "FLUX")  # erg/s/cm^2/Å
    flux ./= h*c ./ λ₀  # convert erg/s to ph/s [ph/s/cm^2/Å]
    flux .*= 1e4  # convert cm^2 to m^2 [ph/s/m^2/Å]
    flux .*= 10  # convert Å to nm [ph/s/m^2/nm]

    if λ == []
        return λ₀, flux
    else
        itp = interpolate((λ₀,), flux, Gridded(Linear()))
        flux = itp(λ)
        return λ, flux
    end
end

function fourier_filter(x, r; FTYPE=Float64)
    ## Keep frequencies inside r
    dim = size(x, 1)
    mask = ones(FTYPE, dim, dim)
    xx = (1:dim) .* ones(dim)'
    xx .-= dim÷2+1
    rr = hypot.(xx, xx')
    mask[rr .> r] .= 0

    X = ft(x)
    x_filtered = real.(ift(X .* mask))
    return x_filtered
end

function filter_to_rmse!(ϕ_smooth, ϕ, rms_target, mask, dim; FTYPE=Float64)
    samples = dim-dim÷4:dim
    rms = zeros(FTYPE, length(samples))
    i = 1
    for n in samples
        ϕ_smooth .= fourier_filter(ϕ, dim-n)
        rms[i] = sqrt(mean((ϕ_smooth[mask .> 0] .- ϕ[mask .> 0]).^2))
        i += 1
    end
    ϕ_smooth .= fourier_filter(ϕ, dim-argmin(abs.(rms .- rms_target)))
end

function setup_fft(dim; FTYPE=Float64)
    container = Matrix{Complex{FTYPE}}(undef, dim, dim)
    pft = plan_fft!(container)
    function fft!(out, in)
        fftshift!(container, in)
        mul!(container, pft, container)
        ifftshift!(out, container)
    end

    return fft!
end

function setup_ifft(dim; FTYPE=Float64)
    scale_ifft = FTYPE(dim)
    container = Matrix{Complex{FTYPE}}(undef, dim, dim)
    pift = plan_ifft!(container)
    function ifft!(out, in)
        ifftshift!(container, in)
        mul!(container, pift, container)
        fftshift!(out, container)
        out .*= scale_ifft
    end

    return ifft!
end

function setup_conv(dim; FTYPE=Float64)
    ft1 = setup_fft(dim, FTYPE=FTYPE)
    ft2 = setup_fft(dim, FTYPE=FTYPE)
    ift1 = setup_ifft(dim, FTYPE=FTYPE)
    container1 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    container2 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    container3 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    function conv!(out, in1, in2)
        ft1(container1, in1)
        ft2(container2, in2)
        container3 .= container1 .* container2
        ift1(out, container3)
    end

    return conv!
end

function setup_corr(dim; FTYPE=Float64)
    ft1 = setup_fft(dim, FTYPE=FTYPE)
    ft2 = setup_fft(dim, FTYPE=FTYPE)
    ift1 = setup_ifft(dim, FTYPE=FTYPE)
    container1 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    container2 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    container3 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    function conv!(out, in1, in2)
        ft1(container1, in1)
        ft2(container2, in2)
        container3 .= container1 .* conj.(container2)
        ift1(out, container3)
    end

    return conv!
end

function setup_operator_mul(dim; FTYPE=Float64)
    container = Matrix{Complex{FTYPE}}(undef, dim, dim)
    function apply!(in, operator)
        mul!(container, operator, in)
        return container
    end

    return apply!
end
