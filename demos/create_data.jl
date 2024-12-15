include("../src/mfbd.jl")
using Main.MFBD

using Statistics
############# Data Parameters #############
FTYPE = Float32;
folder = "data/test"
id = ""
verb = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 32

nsubaps_side = 12
nepochs = 100
nλ = 1
λ_nyquist = 500.0
λ_ref = 500.0
λmin = 500.0
λmax = 500.0
λ = collect(range(λmin, stop=λmax, length=nλ))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)

header = (
    ["WAVELENGTH_START", "WAVELENGTH_END", "WAVELENGTH_STEPS"], 
    [λmin, λmax, nλ], 
    ["Shortest wavelength of mask [nm]", "Largest wavelength of mask [nm]", "Number of wavelength steps"]
)
########## Create Full-Ap Masks ###########
masks_full = Masks(
    dim=image_dim,
    nsubaps_side=1, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
maskfile = "$(folder)/ish_subaps_1x1$(id).fits"
writefits(masks_full.masks, maskfile, header=header)
masks_wfs = Masks(
    dim=image_dim,
    nsubaps_side=nsubaps_side, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
maskfile = "$(folder)/ish_subaps_$(nsubaps_side)x$(nsubaps_side)$(id).fits"
writefits(masks_wfs.masks, maskfile, header=header)
masks = [masks_full, masks_wfs]
# masks = [masks_full]
#########################################

### Detector & Observations Parameters ####
D = 3.6  # m
# fov = 20 * 256 / (132 * (256 / 512))
fov = 20.0
# fov = 100.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full * nsubaps_side
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
# qe = ones(FTYPE, nλ)
rn = 2.0
exptime = 5e-3
noise = false
ζ = 0.0
######### Create Full-Ap Detector #########
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
observations_full = Observations(
    detector_full,
    ζ=ζ,
    D=D,
    nepochs=nepochs,
    nsubaps=1,
    dim=image_dim,
    α=0.5,
    FTYPE=FTYPE
)
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
observations_wfs = Observations(
    detector_wfs,
    ζ=ζ,
    D=D,
    nepochs=nepochs,
    nsubaps=masks_wfs.nsubaps,
    dim=wfs_dim,
    α=0.5,
    FTYPE=FTYPE
)
observations = [observations_full, observations_wfs]
# observations = [observations_full]
###########################################

############ Object Parameters ############
objectfile = "data/star.fits"
~, spectrum = solar_spectrum(λ=λ)
# spectrum = ones(FTYPE, nλ)
template = false
mag = 4
background_mag = Inf ## mag / arcsec^2
flux = mag2flux(λ, spectrum, mag, observations_full.detector, D=D, ζ=ζ, exptime=exptime)
background_flux = mag2flux(λ, ones(nλ), background_mag, observations_full.detector, D=D, ζ=ζ, exptime=exptime)
background_flux *= fov^2
object_height = Inf# 1.434e6  # km
############## Create object ##############
object = Object(
    flux=flux,
    background_flux=background_flux,
    λ=λ,
    fov=fov,
    height=object_height, 
    dim=image_dim,
    spectrum=spectrum,
    qe=qe,
    objectfile=objectfile, 
    template=template,
    FTYPE=FTYPE
)
writefits(object.object, "$(folder)/object_truth$(id).fits", header=header)
###########################################

########## Anisopatch Parameters ##########
isoplanatic = true
patch_overlap = 0.5
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, patch_overlap, isoplanatic=isoplanatic, FTYPE=FTYPE)
###########################################

########## Atmosphere Parameters ##########
l0 = 0.01  # m
L0 = 100.0  # m
Dr0_vertical = 20.0
Dr0_composite = Dr0_vertical * sec(ζ*pi/180)
r0_composite = D / Dr0_composite
heights = [12.5] # [0.0, 7.0, 12.5]
wind_speed = wind_profile_roberts2011(heights, ζ)
wind_direction = [45.0]# [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
propagate = false
r0 = (r0_composite / nlayers^(-3/5)) .* ones(nlayers)  # m
seeds = [713]# [713, 1212, 525118]
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    l0=l0,
    L0=L0,
    r0=r0, 
    wind=wind, 
    heights=heights, 
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ=λ,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    seeds=seeds, 
    FTYPE=FTYPE
)
########## Create phase screens ###########
calculate_screen_size!(atmosphere, observations_full, object, patches, verb=verb)
calculate_pupil_positions!(atmosphere, observations_full, verb=verb)
calculate_layer_masks_eff!(patches, atmosphere, observations_full, object, masks_full, verb=verb)
create_phase_screens!(atmosphere, observations_full)
# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full, 0.1)
atmosphere.opd .*= atmosphere.masks
# opd_smooth .*= atmosphere.masks

# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full)
calculate_composite_pupil_eff(patches, atmosphere, observations, object, masks, build_dim=image_dim, propagate=propagate)
# writefits(atmosphere.masks, "$(folder)/layer_masks$(id).fits", header=header)
# writefits(observations_full.A, "$(folder)/Dr0_$(Dr0_composite)_ISH1x1_amplitude$(id).fits")
# writefits(observations_wfs.A, "$(folder)/Dr0_$(Dr0_composite)_ISH$(nsubaps_side)x$(nsubaps_side)_amplitude$(id).fits")
# writefits(atmosphere.ϕ, "$(folder)/Dr0_$(Dr0_composite)_phase_full$(id).fits", header=header)
# writefits(patches.ϕ_slices, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_phase_slices$(id).fits", header=header)
# writefits(patches.ϕ_composite, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_phase_composite$(id).fits", header=header)
# writefits(atmosphere.opd, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full$(id).fits")
# writefits(opd_smooth, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full_smooth$(id).fits")
# exit()
###########################################

########## Create Full-Ap images ##########
create_images_eff(patches, observations, atmosphere, masks, object, build_dim=image_dim, noise=noise)
outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH1x1_images$(id).fits"
writefits(observations_full.images, outfile, header=header)
outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH$(nsubaps_side)x$(nsubaps_side)_images$(id).fits"
writefits(observations_wfs.images, outfile, header=header)
# outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH1x1_monochromatic_images$(id).fits"
# writefits(observations_full.monochromatic_images, outfile, header=header)
# outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH1x1_psfs$(id).fits"
# writefits(patches.psfs[1], outfile, header=header)
###########################################
