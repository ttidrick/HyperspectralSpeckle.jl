include("../src/mfbd.jl");
using Main.MFBD;
using Statistics

############# Data Parameters #############
FTYPE = Float32;
folder = "data/test";
verb = true
plot = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 32
nsubaps_side = 12
nλ₀ = 1
nλ = 1
nλint = 1
λ_nyquist = 500.0
λ_ref = 500.0
λmin = 500.0
λmax = 500.0
λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
λ₀ = (nλ₀ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ₀))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
###########################################
id = "_slopes"

### Detector & Observations Parameters ####
D = 3.6  # m
fov = 20.0
pixscale_full = fov / image_dim  # 0.25 .* ((λ .* 1e-9) .* 1e6) ./ D
pixscale_wfs = pixscale_full * nsubaps_side
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
rn = 2.0
exptime = 5e-3
ζ = 0.0
######### Create Detector object ##########
filter = Filter(filtername="Bessell:V", FTYPE=FTYPE)
##### Create WFS Observations object ######
detector_wfs = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_wfs,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    filter=filter,
    FTYPE=FTYPE
)
datafile = "$(folder)/Dr0_20_ISH$(nsubaps_side)x$(nsubaps_side)_images.fits"
observations_wfs = Observations(
    detector_wfs,
    ζ=ζ,
    D=D,
    α=0.5,
    nsubaps_side=nsubaps_side,
    datafile=datafile,
    FTYPE=FTYPE
)
###########################################

masks_wfs = Masks(
    dim=image_dim,
    nsubaps_side=nsubaps_side, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)

∇ϕx_vec, ∇ϕy_vec = calculate_wfs_slopes(observations_wfs)
∇ϕx = zeros(eltype(∇ϕx_vec), observations_wfs.nsubaps_side, observations_wfs.nsubaps_side, observations_wfs.nepochs)
∇ϕy = zeros(eltype(∇ϕy_vec), observations_wfs.nsubaps_side, observations_wfs.nsubaps_side, observations_wfs.nepochs)

using GLMakie
for t=1:observations_wfs.nepochs
    ∇ϕx[:, :, t] .= stack2mosaic(∇ϕx_vec[:, t], observations_wfs.nsubaps_side, masks_wfs.ix)
    ∇ϕy[:, :, t] .= stack2mosaic(∇ϕy_vec[:, t], observations_wfs.nsubaps_side, masks_wfs.ix)
end

using FourierTools
acfx = ccorr_psf(∇ϕx, ∇ϕx)
acfy = ccorr_psf(∇ϕy, ∇ϕy)
acf = hypot.(acfx, acfy)
writefits(acf, "$(folder)/acf.fits")
exit()

fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1], aspect=1)
hidedecorations!(ax)
hidespines!(ax)
obs = Observable(rotr90(acf[:, :, 1]))
heatmap!(ax, obs)
record(fig, "$(folder)/acf.mp4", 1:observations_wfs.nepochs; framerate = 15) do t
    obs[] = rotr90(acf[:, :, t])
end

