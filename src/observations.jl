import Interpolations: LinearInterpolation, Line


const FILTER_FILENAMES = Dict(
    "Bessell:U"=>"data/filters/Generic_Bessell.U.dat",
    "Bessell:B"=>"data/filters/Generic_Bessell.B.dat",
    "Bessell:V"=>"data/filters/Generic_Bessell.V.dat",
    "Bessell:R"=>"data/filters/Generic_Bessell.R.dat",
    "Bessell:I"=>"data/filters/Generic_Bessell.I.dat",
    "broadband"=>"data/filters/broadband.dat"
)

struct Filter{T<:AbstractFloat}
    λ::Vector{T}
    response::Vector{T}
    function Filter(;
            filtername="", 
            λ=[],
            response=[],
            FTYPE=Float64
        )
        if filtername != ""
            λfilter, response = readfile(FILTER_FILENAMES["$filtername"])
            if λ != []
                itp = LinearInterpolation(λfilter, response, extrapolation_bc=Line())
                response = itp(λ)
                λfilter = λ
            end
        else
            λfilter = λ
        end

        return new{FTYPE}(λfilter, response)
    end
end

struct Detector{T<:AbstractFloat, S<:Integer}
    λ::Vector{T}
    λ_nyquist::T
    qe::Vector{T}
    rn::T
    gain::T
    saturation::T
    pixscale::T
    exptime::T
    filter::Filter{T}
    function Detector(;
            λ=[],
            λ_nyquist=400.0,
            pixscale=0.0,
            qe=[1.0],
            rn=0.0,
            gain=1.0,
            saturation=1e99,
            exptime=5e-3,
            filter=Filter(λ=λ, response=ones(FTYPE, length(λ)), FTYPE=Float64),
            FTYPE=Float64,
            DTYPE=UInt16
        )
        return new{FTYPE, DTYPE}(λ, λ_nyquist, qe, rn, gain, saturation, pixscale, exptime, filter)
    end
end

mutable struct Observations{T<:AbstractFloat}
    detector::Detector{T}
    ζ::T
    D::T
    nepochs::Int64
    nsubaps::Int64
    α::T
    dim::Int64
    images::Array{T, 4}
    entropy::Matrix{T}
    model_images::Array{T, 4}
    psfs::Array{T, 5}
    monochromatic_images::Array{T, 5}
    w::Vector{Int64}
    function Observations(
            detector;
            ζ=0.0,
            D=1.0,
            nepochs=1,
            nsubaps=1,
            α=1.0,
            dim=256,
            datafile::String = "",
            FTYPE=Float64
        )
        if (datafile != "")
            images, nsubaps, nepochs, dim = readimages(datafile, FTYPE=FTYPE)
            entropy = [calculate_entropy(images[:, :, n, t]) for n=1:nsubaps, t=1:nepochs]
            println("Loading $(nepochs) frames of size $(dim)x$(dim) pixels for $(nsubaps) subapertures")
            print(" |-> "); printstyled("$(datafile)\n", color=:red)
            return new{FTYPE}(detector, ζ, D, nepochs, nsubaps, α, dim, images, entropy)
        else
            return new{FTYPE}(detector, ζ, D, nepochs, nsubaps, α, dim)
        end
    end
end
