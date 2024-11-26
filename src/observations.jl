import Interpolations: LinearInterpolation, Line


const ELEMENT_FILENAMES = Dict(
    ## Filter
    "Bessell:U"=>"data/optics/Generic_Bessell.U.dat",
    "Bessell:B"=>"data/optics/Generic_Bessell.B.dat",
    "Bessell:V"=>"data/optics/Generic_Bessell.V.dat",
    "Bessell:R"=>"data/optics/Generic_Bessell.R.dat",
    "Bessell:I"=>"data/optics/Generic_Bessell.I.dat",
    ## Lens Coatings
    "Thorlabs:A"=>"data/optics/thorlabs-A.dat",
    "Thorlabs:B"=>"data/optics/thorlabs-B.dat",
    "Thorlabs:AB"=>"data/optics/thorlabs-AB.dat",
    "Thorlabs:MLA-AR"=>"data/optics/thorlabs-mla-ar.dat",
    ## Mirror Coatings
    "Thorlabs:OAP-P01"=>"data/optics/thorlabs-OAP-45AOI-P01.dat",
    "Thorlabs:Plano-P01"=>"data/optics/thorlabs-plano-45AOI-P01.dat",
    ## Dichroics
    "Thorlabs:DMLP650P-transmitted"=>"data/optics/DMLP650-transmitted.dat",
    "Thorlabs:DMLP650P-reflected"=>"data/optics/DMLP650-reflected.dat",
    "Thorlabs:DMLP805P-transmitted"=>"data/optics/DMLP805-transmitted.dat",
    "Thorlabs:DMLP805P-reflected"=>"data/optics/DMLP805-reflected.dat",
)

struct OpticalElement{T<:AbstractFloat}
    name::String
    λ::Vector{T}
    response::Vector{T}
    xflip::Bool
    yflip::Bool
    function OpticalElement(;
            λ=[],
            response=[],
            xflip=false,
            yflip=false,
            name="",
            FTYPE=Float64
        )
        if name != ""
            λ₀, response = readfile(ELEMENT_FILENAMES["$name"])
            if λ != []
                itp = LinearInterpolation(λ₀, response, extrapolation_bc=Line())
                response = itp(λ)
            else
                λ = λ₀
            end
        end

        return new{FTYPE}(name, λ, response, xflip, yflip)
    end
end

struct Detector{T<:AbstractFloat, S<:Real}
    λ::Vector{T}
    λ_nyquist::T
    qe::Vector{T}
    rn::T
    gain::T
    saturation::T
    pixscale::T
    exptime::T
    function Detector(;
            λ=[],
            λ_nyquist=400.0,
            pixscale=0.0,
            qe=[1.0],
            rn=0.0,
            gain=1.0,
            saturation=1e99,
            exptime=5e-3,
            verb=true,
            FTYPE=Float64,
            DTYPE=FTYPE
        )
        if verb == true
            print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Detector\n"); print(Crayon(reset=true))
            println("\tBit Depth: $(DTYPE)")
            println("\tRN: $(rn) e⁻")
            println("\tGain: $(gain) e⁻/ADU")
            println("\tSaturation: $(saturation)")
            println("\tExposure time: $(exptime) s")
            println("\tPlate Scale: $(pixscale) arcsec/pix")
            println("\tWavelength: $(minimum(λ))—$(maximum(λ)) nm")
            println("\tNyquist sampled wavelength: $(λ_nyquist) nm")
        end
        return new{FTYPE, DTYPE}(λ, λ_nyquist, qe, rn, gain, saturation, pixscale, exptime)
    end
end

struct OpticalSystem{T<:AbstractFloat}
    elements::Vector{OpticalElement{T}}
    λ::Vector{T}
    response::Vector{T}
    xflip::Bool
    yflip::Bool
    function OpticalSystem(
            elements,
            λ;
            verb=true,
            FTYPE=Float64
        )
        if verb == true
            print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Optical system\n"); print(Crayon(reset=true))
            println("\tNumber of elements: $(length(elements))")
            println("\tWavelength: $(minimum(λ))—$(maximum(λ)) nm")
        end

        xflip = false
        yflip = false
        response = ones(FTYPE, length(λ))
        for element in elements
            itp = LinearInterpolation(element.λ, element.response, extrapolation_bc=Line())
            response .*= itp(λ)
            xflip = xflip ⊻ element.xflip
            yflip = yflip ⊻ element.yflip
        end

        return new{FTYPE}(elements, λ, response, xflip, yflip)
    end
end

mutable struct Observations{T<:AbstractFloat, S<:Real}
    optics::OpticalSystem{T}
    phase_static::Array{T, 3}
    detector::Detector{T, S}
    ζ::T
    D::T
    nepochs::Int64
    nsubaps::Int64
    nsubaps_side::Int64
    dim::Int64
    images::Array{S, 4}
    entropy::Matrix{T}
    model_images::Array{T, 4}
    monochromatic_images::Array{T, 5}
    w::Vector{Int64}
    function Observations(
            optics,
            detector;
            ζ=Inf,
            D=Inf,
            nepochs=0,
            nsubaps=0,
            nsubaps_side=1,
            dim=0,
            ϕ_static=[;;;],
            datafile::String="",
            verb=true,
            FTYPE=Float64
        )
        DTYPE = gettypes(detector)[2]
        optics.response .*= detector.qe
        if (datafile != "")
            images, nsubaps, nepochs, dim = readimages(datafile, FTYPE=FTYPE)
            entropy = [calculate_entropy(images[:, :, n, t]) for n=1:nsubaps, t=1:nepochs]
            if verb == true
                print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Observations\n"); print(Crayon(reset=true))
                print("\tFile: "); print(Crayon(foreground=:red), "$(datafile)\n"); print(Crayon(reset=true));
                println("\tImage Size: $(dim)x$(dim) pixels")
                println("\tNumber of frames: $(nepochs)")
                println("\tNumber of subapertures: $(nsubaps_side)×$(nsubaps_side) subapertures")
                println("\tTelescope Diameter: $(D) m")
                println("\tZenith angle: $(ζ) deg")
            end
            return new{FTYPE, DTYPE}(optics, ϕ_static, detector, ζ, D, nepochs, nsubaps, nsubaps_side, dim, images, entropy)
        else
            if verb == true
                print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Observations\n"); print(Crayon(reset=true))
                println("\tImage Size: $(dim)x$(dim) pixels")
                println("\tNumber of frames: $(nepochs)")
                println("\tNumber of subapertures: $(nsubaps_side)×$(nsubaps_side) subapertures")
                println("\tTelescope Diameter: $(D) m")
                println("\tZenith angle: $(ζ) deg")
            end
            return new{FTYPE, DTYPE}(optics, ϕ_static, detector, ζ, D, nepochs, nsubaps, nsubaps_side, dim)
        end
    end
end

@views function calculate_wfs_slopes(observations_wfs)
    FTYPE = gettype(observations_wfs)
    ~, ~, nsubaps, nepochs = size(observations_wfs.images)
    ∇ϕx = zeros(FTYPE, nsubaps, nepochs)
    ∇ϕy = zeros(FTYPE, nsubaps, nepochs)
    composite_image = dropdims(sum(observations_wfs.images, dims=(3, 4)), dims=(3, 4))
    for n=1:nsubaps
        for t=1:nepochs
            # Δy, Δx = Tuple(argmax(ccorr_psf(composite_image, observations_wfs.images[:, :, n, t])))
            Δx, Δy = center_of_gravity(observations_wfs.images[:, :, n, t])
            ∇ϕx[n, t] = Δx * observations_wfs.D / observations_wfs.nsubaps_side
            ∇ϕy[n, t] = Δy * observations_wfs.D / observations_wfs.nsubaps_side
        end
    end
    return ∇ϕx, ∇ϕy
end
