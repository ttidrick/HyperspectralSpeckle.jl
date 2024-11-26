include("../src/mfbd.jl");
using Main.MFBD;
using Statistics;
using LuckyImaging;
show_the_sausage()

############# Data Parameters #############
FTYPE = Float32;
folder = "/home/dan/Desktop/ApJL_2024/separations/recon/Delta_mag10";
verb = true
plot = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64
nsubaps_side = 6
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
id = "_aniso"

########## Anisopatch Parameters ##########
## Unused but sets the size of the layer ##
isoplanatic = false
patch_overlap = 0.5
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, patch_overlap, isoplanatic=isoplanatic, FTYPE=FTYPE)
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
# fov = 256 / 5 * 2
# fov = 256 / 78.5 * 20
# fov = 20 * 256 / (132 * (256 / 512))
fov = 20.0
pixscale_full = fov / image_dim  # 0.25 .* ((λ .* 1e-9) .* 1e6) ./ D
pixscale_wfs = pixscale_full * nsubaps_side
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
# qe = ones(FTYPE, nλ)
rn = 2.0
exptime = 5e-3
ζ = 0.0
######### Create Detector object ##########
filter = Filter(filtername="Bessell:V", FTYPE=FTYPE)
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    filter=filter,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
datafile = "$(folder)/Dr0_20_ISH1x1_images.fits"
observations_full = Observations(
    detector_full,
    ζ=ζ,
    D=D,
    α=0.5,
    datafile=datafile,
    FTYPE=FTYPE
)
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
datafile = "$(folder)/Dr0_20_ISH6x6_images.fits"
observations_wfs = Observations(
    detector_wfs,
    ζ=ζ,
    D=D,
    α=0.5,
    datafile=datafile,
    FTYPE=FTYPE
)
# observations = [observations_wfs, observations_full]
observations = [observations_full]
###########################################

########### Load Full-Ap Masks ############
masks_full = Masks(
    dim=observations_full.dim,
    nsubaps_side=1, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
############ Create WFS Masks #############
masks_wfs = Masks(
    dim=observations_full.dim,
    nsubaps_side=nsubaps_side, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
# masks = [masks_wfs, masks_full]
masks = [masks_full]
###########################################

############ Object Parameters ############
object_height = Inf# 1.434e6  # km
############## Create object ##############
object = Object(
    λ=λ,
    height=object_height, 
    fov=fov,
    dim=observations_full.dim,
    FTYPE=FTYPE
)

# all_subap_images = block_replicate(dropdims(sum(observations_wfs.images, dims=(3, 4)), dims=(3, 4)), image_dim) .+ dropdims(sum(observations_full.images, dims=(3, 4)), dims=(3, 4))
# all_subap_images = lucky_image(observations_full.images[:, :, 1, :], dims=3, q=0.9)
# object.object = repeat(all_subap_images, 1, 1, nλ)
# object.object ./= sum(object.object)
# object.object .*= 2 * mean(sum(observations_full.images, dims=(1, 2, 3)))
# object.object .-= mean(object.object)
object.object = readfits("$(folder)/object_recon_aniso.fits")
# object.object = readfits("$(folder)/object_truth_2arcsec.fits")
###########################################

########## Atmosphere Parameters ##########
heights = [0.0, 7.0, 12.5]
wind_speed = wind_profile_roberts2011(heights, ζ)
# heights .*= 0.0
heights .= [0.0, 5.0, 10.0]
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
scaleby_wavelength = λ_nyquist ./ λ
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    wind=wind, 
    heights=heights, 
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ=λ,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    FTYPE=FTYPE
)
########## Create phase screens ###########
calculate_screen_size!(atmosphere, observations_full, object, patches)
calculate_pupil_positions!(atmosphere, observations_full)
calculate_layer_masks_eff!(patches, atmosphere, observations_full, object, masks_full)
# atmosphere.opd = zeros(FTYPE, size(atmosphere.masks)[1:3])
atmosphere.opd = readfits("$(folder)/opd_recon_aniso.fits")
# atmosphere.opd = readfits("$(folder)/Dr0_20_opd_full_2arcsec.fits")
###########################################

######### Reconstruction Object ###########
reconstruction = Reconstruction(
    atmosphere,
    observations,
    object,
    patches,
    λmin=λmin,
    λmax=λmax,
    nλ=nλ,
    nλint=nλint,
    niter_mfbd=100,
    maxiter=30,
    # indx_boot=[1:2],
    weight_function=mixed_weighting,
    gradient_object=gradient_object_mixednoise!,
    gradient_opd=gradient_opd_mixednoise!,
    maxeval=Dict("opd"=>1000, "object"=>1000),
    smoothing=false,
    # minFWHM=0.5,
    # maxFWHM=0.5,
    # fwhm_schedule=ReciprocalSchedule(5.0, 0.5),
    # fwhm_schedule=ConstantSchedule(0.5),
    grtol=0.0,
    xrtol=0.0,
    build_dim=image_dim,
    verb=verb,
    plot=plot,
    FTYPE=FTYPE
);
reconstruct_blind!(reconstruction, observations, atmosphere, object, masks, patches, write=true, folder=folder, id=id)
###########################################

# calculate_composite_pupil_eff(patches, atmosphere, observations, object, masks, build_dim=image_dim, propagate=false)
# writefits(patches.ϕ_slices, "$(folder)/phase_recon_slices$(id).fits")
# writefits(patches.ϕ_composite, "$(folder)/phase_recon_composite$(id).fits")
# writefits(atmosphere.masks, "$(folder)/layer_masks$(id).fits")
# writefits(atmosphere.opd, "$(folder)/opd_recon_planesubtracted$(id).fits")

###########################################
## Write isoplanatic phases and images ####
# writefits(observations_full.model_images, "$(folder)/models_ISH1x1_recon$(id).fits")
# writefits(observations_wfs.model_images, "$(folder)/models_ISH6x6_recon$(id).fits")
# writefits(object.object, "$(folder)/object_recon$(id).fits")
# writefits(atmosphere.opd, "$(folder)/opd_recon_planesubtracted$(id).fits")
# writefile([reconstruction.ϵ], "$(folder)/recon$(id).dat")
###########################################
