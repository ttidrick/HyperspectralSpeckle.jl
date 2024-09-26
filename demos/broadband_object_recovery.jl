include("../src/mfbd.jl");
using Main.MFBD;
using Statistics;
using LuckyImaging;
show_the_satellite()

############# Data Parameters #############
FTYPE = Float32;
folder = "/home/dan/Desktop/JASS_2024/prime-95b/zenith17";
verb = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 512
nλ = 101
nsubλ = 1
nλtotal = (nλ - 1) * nsubλ + 1
λ_nyquist = 400.0
λ_ref = 500.0
λmin = 400.0
λmax = 1000.0
λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
λtotal = (nλtotal == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλtotal))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
Δλtotal = (nλtotal == 1) ? 1.0 : (λmax - λmin) / (nλtotal - 1)
###########################################
# id = "_nlambda$(nλ)"
id = "_grey"

########## Anisopatch Parameters ##########
## Unused but sets the size of the layer ##
isoplanatic = true
patch_overlap = 0.5
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, patch_overlap, isoplanatic=isoplanatic, FTYPE=FTYPE)
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
fov = 8.0  # arcsec
pixscale_full = fov / image_dim  # 0.25 .* ((λ .* 1e-9) .* 1e6) ./ D
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λtotal)
# qe = ones(FTYPE, nλ)
rn = 1.0
exptime = 5e-3
ζ = 17.0
######### Create Detector object ##########
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    λ=λtotal,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
datafile = "$(folder)/Dr0_20_ISH1x1_images.fits"
observations_full = Observations(
    detector_full,
    ζ=ζ,
    D=D,
    α=1.0,
    datafile=datafile,
    FTYPE=FTYPE
)
##### Create WFS Observations object ######
observations = [observations_full]
[observations[dd].A = ones(FTYPE, image_dim, image_dim, observations[dd].nsubaps, patches.npatches, observations[dd].nepochs, nλtotal) for dd=1:length(observations)]
###########################################

########### Load Full-Ap Masks ############
masks_full = Masks(
    dim=observations_full.dim,
    nsubaps_side=1, 
    λ=λtotal,
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
masks = [masks_full]
###########################################

############ Object Parameters ############
object_height = 515.0  # km
############## Create object ##############
object = Object(
    λ=λ,
    height=object_height, 
    fov=fov,
    dim=observations_full.dim,
    FTYPE=FTYPE
)
# if nλ == 1
    # all_subap_images = lucky_image(observations_full.images[:, :, 1, :], dims=3, q=0.9)
    # object.object = repeat(all_subap_images, 1, 1, nλ)
    # object.object ./= sum(object.object)
    # object.object .*= mean(sum(observations_full.images, dims=(1, 2)), dims=(3, 4))
    object.object = zeros(FTYPE, image_dim, image_dim, nλ)
    # elseif nλ > 1 && nλ₀ == 1
#     object.object = repeat(readfits("$(folder)/object_recon_aniso_nlambda$(nλ₀).fits", FTYPE=FTYPE), 1, 1, nλ)
# elseif nλ > 1 && nλ₀ > 1
    # nλ₀ = 21
    # λ₀ = (nλ₀ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ₀))
    # object.object = interpolate_object(readfits("$(folder)/object_recon_aniso_nlambda$(nλ₀)_grey.fits", FTYPE=FTYPE), λ₀, λ)
    # object.object = interpolate_object(readfits("$(folder)/Dr0_20_opd_full_smooth.fits", FTYPE=FTYPE), λ₀, λ)
    # object.object = interpolate_object(readfits("$(folder)/hyperspectral_sat.fits", FTYPE=FTYPE), λ₀, λ)
# end
# object.object .+= 100 .* rand(FTYPE, image_dim, image_dim, nλ)
###########################################

########## Atmosphere Parameters ##########
heights = [0.0, 7.0, 12.5]
wind_speed = wind_profile_roberts2011(heights, ζ)
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
scaleby_wavelength = λ_nyquist ./ λtotal
Dmeta = D .+ (fov/206265) .* (heights .* 1000)
sampling_nyquist_mperpix = (2*D / image_dim) .* ones(nlayers)
sampling_nyquist_arcsecperpix = (fov / image_dim) .* (Dmeta ./ D)
# maskfile = "$(folder)/layer_masks.fits"
############ Create Atmosphere ############
atmosphere = Atmosphere(
    wind=wind, 
    heights=heights, 
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ=λtotal,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    # maskfile=maskfile,
    verb=verb,
    FTYPE=FTYPE
)
########## Create phase screens ###########
calculate_screen_size!(atmosphere, observations_full, object, patches, verb=verb)
calculate_pupil_positions!(atmosphere, observations_full, verb=verb)
calculate_layer_masks_eff_alt!(atmosphere, observations[end], object, masks[end], verb=verb)
# atmosphere.opd = readfits("$(folder)/opd_recon_aniso_nlambda101_grey.fits", FTYPE=FTYPE)
atmosphere.opd = readfits("$(folder)/Dr0_20_opd_full.fits", FTYPE=FTYPE)
# atmosphere.opd .+ 10 .* randn(size(atmosphere.opd)...)
# atmosphere.opd = randn(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
###########################################

######### Reconstruction Object ###########
# regularizers = Regularizers(
#     # βλ=1000,
#     # βo=10,
#     # λ_reg=λtv_reg(image_dim, λ, FTYPE=FTYPE),
#     # o_reg=tv2_reg(image_dim, FTYPE=FTYPE),
#     # o_reg=l2_reg(FTYPE=FTYPE),
#     FTYPE=FTYPE  
# )
reconstruction = Reconstruction(
    atmosphere,
    observations,
    object,
    patches,
    niter_mfbd=1,
    maxiter=2000,
    # weight_function=gaussian_weighting,
    maxeval=Dict("opd"=>1, "object"=>10000),
    smoothing=false,
    grtol=1e-6,
    # regularizers=regularizers,
    grey=true,
    build_dim=image_dim,
    verb=verb,
    FTYPE=FTYPE
);
opd_solve!(reconstruction, observations, atmosphere, object, masks, patches, verb="silent", plot=false)
object_solve!(reconstruction, observations, object, patches, verb="full", plot=true)

###########################################
## Write isoplanatic phases and images ####
writefits(observations_full.model_images, "$(folder)/models_ISH1x1_recon$(id).fits")
# writefits(observations_full.psfs, "$(folder)/psfs_ISH1x1_recon_aniso$(id).fits")
writefits(object.object, "$(folder)/object_recon$(id).fits")
# writefits(atmosphere.opd, "$(folder)/opd_recon_aniso$(id).fits")
writefile(reconstruction.ϵ, "$(folder)/recon$(id).dat")
###########################################