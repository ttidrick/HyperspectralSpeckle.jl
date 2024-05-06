include("../src/mfbd.jl");
using Main.MFBD;
using Statistics;
using LuckyImaging;
show_the_sausage()

############# Data Parameters #############
FTYPE = Float32;
folder = "data/MonoISH/2arcsec";
id = "_iso"
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
nsubaps_side = 6
nλ = 1
λ_nyquist = 500.0
λ_ref = 500.0
λmin = 500.0
λmax = 500.0
λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
###########################################

########## Anisopatch Parameters ##########
## Unused but sets the size of the layer ##
isoplanatic = true
patch_overlap = 0.5
patch_dim = 128
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, patch_overlap, isoplanatic=isoplanatic, FTYPE=FTYPE)
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
pixscale_full = 0.25 .* ((λ .* 1e-9) .* 1e6) ./ D
pixscale_wfs = pixscale_full .* nsubaps_side
qefile = "data/zyla_qe.dat"
~, qe = readqe(qefile, λ=λ)
rn = 1.0
exptime = 5e-3
ζ = 0.0
######### Create Detector object ##########
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    λ=λ,
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
    filtername="Bessell:V",
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
observations = [observations_wfs, observations_full]
[observations[dd].A = ones(FTYPE, image_dim, image_dim, observations[dd].nsubaps, patches.npatches, observations[dd].nepochs, nλ) for dd=1:length(observations)]
# observations[1].A = readfits("$(folder)/Dr0_20_ISH6x6_amplitude.fits", FTYPE=FTYPE)
# observations[2].A = readfits("$(folder)/Dr0_20_ISH1x1_amplitude.fits", FTYPE=FTYPE)
# observations = [observations_full]
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
nsubaps_side = 6
masks_wfs = Masks(
    dim=observations_full.dim,
    nsubaps_side=nsubaps_side, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
masks = [masks_wfs, masks_full]
# masks = [masks_full]
###########################################

############ Object Parameters ############
object_height = Inf32  # km
# object_size = 13.0  # m
# fov = 206265 * object_size / (object_height*1e3)  # arcsec
# fov = 30.0  # arcsec
fov = 2 * image_dim/sqrt(55^2 + 55^2)
############## Create object ##############
object = Object(
    λ=λ,
    height=object_height, 
    fov=fov,
    dim=observations_full.dim,
    FTYPE=FTYPE
)
# all_subap_images = lucky_image(observations_full.images[:, :, 1, :], dims=3, q=0.9)
# object.object = repeat(all_subap_images, 1, 1, nλ)
object.object = repeat(mean(observations_full.images[:, :, 1, :], dims=3), 1, 1, nλ)
# object.object = readfits("$(folder)/object_recon_aniso_at_recovered_heights_smoothed.fits", FTYPE=FTYPE)
###########################################

########## Atmosphere Parameters ##########
heights = [0.0, 7.0, 12.5]
wind_speed = wind_profile_roberts2011(heights, ζ)
heights .*= 0
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
scaleby_wavelength = λ_nyquist ./ λ
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
    λ=λ,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    # maskfile=maskfile,
    FTYPE=FTYPE
)
########## Create phase screens ###########
calculate_screen_size!(atmosphere, observations_full, object, patches)
calculate_pupil_positions!(atmosphere, observations_full)
calculate_layer_masks_eff_alt!(atmosphere, observations[end], object, masks[end])
# atmosphere.opd = readfits("$(folder)/opd_recon_aniso_at_recovered_heights_smoothed.fits", FTYPE=FTYPE)
atmosphere.opd = randn(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
###########################################
# change_heights!(patches, atmosphere, object, observations, masks, [0.0, 6.0, 11.0])

######### Reconstruction Object ###########
reconstruction = Reconstruction(
    atmosphere,
    observations,
    object,
    patches,
    niter_mfbd=10,
    maxiter=10,
    smoothing=true,
    # indx_boot=[1:2],
    FTYPE=FTYPE
);

## Reconstruct the layers and the object
runtime = @elapsed reconstruct_blind!(
    reconstruction,
    observations,
    atmosphere,
    object,
    masks,
    patches,
    plot=true,
    verb="full"
);
###########################################
atmosphere.opd .*= atmosphere.masks[:, :, :, 1]
writefits(atmosphere.opd, "$(folder)/opd_recon$(id).fits")
writefits(patches.ϕ_composite, "$(folder)/phase_composite_recon$(id).fits")
for l=1:atmosphere.nlayers
    atmosphere.opd[findall(atmosphere.masks[:, :, l, 1] .> 0), l] .-= mean(atmosphere.opd[findall(atmosphere.masks[:, :, l, 1] .> 0), l])
    atmosphere.opd[:, :, l] .-= fit_plane(atmosphere.opd[:, :, l], atmosphere.masks[:, :, l, 1])
end
calculate_composite_phase_eff!(patches.ϕ_composite, patches.ϕ_slices, patches, atmosphere, observations[end], reconstruction.patch_helpers.extractor)
writefits(atmosphere.opd, "$(folder)/opd_recon_woplane$(id).fits")
writefits(patches.ϕ_composite, "$(folder)/phase_composite_recon_woplanes$(id).fits")
writefits(observations_full.model_images, "$(folder)/models_ISH1x1_recon$(id).fits")
writefits(patches.psfs[2], "$(folder)/psfs_ISH1x1_recon$(id).fits")
writefits(observations_wfs.model_images, "$(folder)/models_ISH6x6_recon$(id).fits")
writefits(patches.psfs[1], "$(folder)/psfs_ISH6x6_recon$(id).fits")
writefits(object.object, "$(folder)/object_recon$(id).fits")
writefile(reconstruction.ϵ, "$(folder)/recon$(id).dat", header="runtime = $(runtime) s")
###########################################