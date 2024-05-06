include("../src/mfbd.jl");
using Main.MFBD;
using Statistics;
using LuckyImaging;
show_the_satellite()

############# Data Parameters #############
FTYPE = Float32;
folder = "data";
datafolder = "amos/curves"
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256

nλ₀ = 101
nλ = 101
λ_nyquist = 400.0
λ_ref = 500.0
λmin = 400.0
λmax = 1000.0
λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
λ₀ = (nλ₀ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ₀))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
Δλ₀ = (nλ₀ == 1) ? 1.0 : (λmax - λmin) / (nλ₀ - 1)

########### Load Full-Ap Masks ############
masks = [Masks(
    dim=image_dim,
    nsubaps_side=1, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)]
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
pixscale = 0.25 .* ((λ .* 1e-9) .* 1e6) ./ (D/2)
qefile = "data/zyla_qe.dat"
~, qe = readqe(qefile, λ=λ)
# qe = ones(FTYPE, nλ)
rn = 1.0
exptime = 5e-3
ζ = 17.0
nsubaps = 1
######### Create Detector object ##########
detector = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
datafile = "$(folder)/Dr0_20_ISH1x1_images.fits"
observations = [Observations(
    detector,
    ζ=ζ,
    D=D,
    α=1,
    datafile=datafile,
    FTYPE=FTYPE
)]
###########################################

############ Object Parameters ############
object_height = 300.0  # km
object_size = 50.0  # m
############## Create object ##############
object = Object(
    λ=λ,
    height=object_height, 
    object_size=object_size,
    dim=observations[1].dim,
    FTYPE=FTYPE
)
###########################################

########## Atmosphere Parameters ##########
heights = [0.0, 7.0, 12.5]
# wind_speed = wind_profile_greenwood(heights, ζ)
wind_speed = wind_profile_roberts2011(heights, ζ)
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
sampling_nyquist = D/(image_dim/2) .* (1 .- heights ./ object_height)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    wind=wind, 
    heights=heights, 
    sampling_nyquist=sampling_nyquist,
    λ=λ,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    FTYPE=FTYPE
)
########## Create phase screens ###########
calculate_screen_size!(atmosphere, observations[1], object)
calculate_pupil_positions!(atmosphere, observations[1])
###########################################

########## Regularizer Object #############
regularizers = Regularizers(
    FTYPE=FTYPE
)
######### Reconstruction Object ###########
reconstruction = Reconstruction(
    observations,
    λ=λ,
    build_dim=observations[1].dim, 
    niter_mfbd=1,
    maxiter=500,
    ndatasets=length(observations), 
    indx_boot=[1:1],
    gtol=(0.0, 1e-9),
    ϕ_smoothing=false,
    regularizers=regularizers,
    FTYPE=FTYPE
);
# if nλ == 1
#     object.object = repeat(lucky_image(observations[1].images[:, :, 1, :], dims=3, q=0.9), 1, 1, nλ)
#     atmosphere.opd = randn(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
# elseif (nλ₀ == 1) && (nλ > 1)
#     object.object = repeat(readfits("$(folder)/object_recon_nlambda1.fits", FTYPE=FTYPE), 1, 1, nλ)
#     # object.object = zeros(FTYPE, observations[1].dim, observations[1].dim, nλ)
#     atmosphere.opd = readfits("$(folder)/opd_recon_nlambda$(nλ₀).fits", FTYPE=FTYPE)
# elseif (nλ₀ > 1) && (nλ > 1)
#     object.object = interpolate_object(readfits("$(folder)/object_recon_nlambda$(nλ₀).fits", FTYPE=FTYPE), λ₀, λ)
#     atmosphere.opd = readfits("$(folder)/opd_recon_nlambda$(nλ₀).fits", FTYPE=FTYPE)
# end

atmosphere.opd = readfits("$(folder)/Dr0_20_opd_full.fits", FTYPE=FTYPE)
object.object = interpolate_object(readfits("$(folder)/hyperspectral_sat.fits", FTYPE=FTYPE), λ₀, λ)
atmosphere.A = ones(FTYPE, reconstruction.build_dim, reconstruction.build_dim, observations[1].nepochs, nλ)
###########################################

############ Helpers Object ###############
helpers = Helpers(
    atmosphere, 
    observations, 
    reconstruction,
    object,
);
###########################################

## Reconstruct the layers and the object
reconstruct_blind!(
    reconstruction,
    observations,
    atmosphere,
    masks,
    helpers,
    regularizers,
    plot=false,
    verb="full"
);
###########################################

id = "_nlambda$(nλ)_phase_update"
## Plot and write the reconstructed object and layers
writefits(observations[1].model_images, "$(folder)/$(datafolder)/models_ISH1x1_recon$(id).fits")
writefits(reconstruction.o_recon, "$(folder)/$(datafolder)/object_recon$(id).fits")
writefits(reconstruction.ϕ_recon, "$(folder)/$(datafolder)/phase_recon$(id).fits")
writefile(reconstruction.ϵ, "$(folder)/$(datafolder)/recon$(id).dat")
