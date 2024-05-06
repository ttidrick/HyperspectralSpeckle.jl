include("../src/mfbd.jl")
using Main.MFBD

############# Data Parameters #############
FTYPE = Float32;
folder = "data/MonoISH/2arcsec";
id = ""
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64

nsubaps_side = 6
nepochs = 31
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
# writefits(masks_full.masks, maskfile, header=header)
############ Create WFS Masks #############
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
# writefits(masks_wfs.masks, maskfile, header=header)
masks = [masks_wfs, masks_full]
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
pixscale_full = 0.25 .* ((λ .* 1e-9) .* 1e6) ./ D  # arcsec / m
pixscale_wfs = pixscale_full .* nsubaps_side
qefile = "data/zyla_qe.dat"
~, qe = readqe(qefile, λ=λ)
rn = 2.0
exptime = 5e-3
noise = false
ζ = 0.0
######### Create Full-Ap Detector #########
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    filtername="Bessell:V",
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
########### Create WFS Detector ###########
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
#### Create WFS Observations object ######
observations_wfs = Observations(
    detector_wfs,
    ζ=ζ,
    D=D,
    nepochs=nepochs,
    nsubaps=nsubaps,
    dim=wfs_dim,
    α=0.5,
    FTYPE=FTYPE
)
observations = [observations_wfs, observations_full]
###########################################

############ Object Parameters ############
objectfile = "data/binary.fits"
~, spectrum = solar_spectrum(λ=λ)
template = false
mag = 4.0
flux = mag2flux(λ, spectrum, mag, observations_full.detector, D=D, ζ=ζ, exptime=exptime)
object_height = Inf32  # km
# object_size = 13.0  # m
# fov = 206265 * object_size / (object_height*1e3)
# fov = 30.0
fov = 2 * image_dim/sqrt(55^2 + 55^2)
############## Create object ##############
object = Object(
    flux=flux, 
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
# writefits(object.object, "$(folder)/hyperspectral_sat$(id).fits", header=header)
###########################################

########## Anisopatch Parameters ##########
isoplanatic = false
patch_overlap = 0.5
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, patch_overlap, isoplanatic=isoplanatic, FTYPE=FTYPE)
###########################################

########## Atmosphere Parameters ##########
l0 = 0.01  # m
L0 = 100.0  # m
Dr0_composite = 20
r0_composite = D / Dr0_composite
heights = [0.0, 7.0, 12.5]
wind_speed = wind_profile_roberts2011(heights, ζ)
# heights .*= 0
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
propagate = false
r0 = (r0_composite / nlayers^(-3/5)) .* ones(nlayers)  # m
seeds = [713, 1212, 525118]
Dmeta = D .+ (fov/206265) .* (heights .* 1000)
sampling_nyquist_mperpix = (2*D / image_dim) .* ones(nlayers)
sampling_nyquist_arcsecperpix = (fov / image_dim) .* (D ./ Dmeta)
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
calculate_screen_size!(atmosphere, observations_full, object, patches)
calculate_pupil_positions!(atmosphere, observations_full)
calculate_layer_masks_eff_alt!(atmosphere, observations[end], object, masks[end])
create_phase_screens!(atmosphere, observations_full)
# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full)
calculate_composite_pupil_eff(patches, atmosphere, observations, object, masks, propagate=propagate)

writefits(atmosphere.masks, "$(folder)/layer_masks.fits", header=header)
# writefits(observations_full.A, "$(folder)/Dr0_$(Dr0_composite)_ISH1x1_amplitude$(id).fits")
# writefits(observations_wfs.A, "$(folder)/Dr0_$(Dr0_composite)_ISH$(nsubaps_side)x$(nsubaps_side)_amplitude$(id).fits")
writefits(atmosphere.ϕ, "$(folder)/Dr0_$(Dr0_composite)_phase_full$(id).fits", header=header)
# writefits(patches.ϕ_slices, "$(folder)/Dr0_$(Dr0_composite)_phase_slices$(id).fits", header=header)
writefits(patches.ϕ_composite, "$(folder)/Dr0_$(Dr0_composite)_phase_composite$(id).fits", header=header)
writefits(atmosphere.opd, "$(folder)/Dr0_$(Dr0_composite)_opd_full$(id).fits")
# writefits(opd_smooth, "$(folder)/$(outfolder)/Dr0_$(Dr0)_opd_full_smooth$(id).fits")
###########################################
# change_heights!(patches, atmosphere, object, observations, masks, heights .* 0)
# writefits(atmosphere.opd, "$(folder)/Dr0_$(Dr0_composite)_opd_full_at_pupil.fits")
# exit()
########## Create Full-Ap images ##########
observations_full.images, psfs_full = create_images_eff(patches, observations_full, atmosphere, masks_full, object, build_dim=image_dim, noise=noise)
outfile = "$(folder)/Dr0_$(Dr0_composite)_ISH1x1_images$(id).fits"
writefits(observations_full.images, outfile, header=header)
outfile = "$(folder)/Dr0_$(Dr0_composite)_ISH1x1_psfs$(id).fits"
writefits(psfs_full, outfile, header=header)
############ Create WFS images ############
observations_wfs.images, psfs_wfs = create_images_eff(patches, observations_wfs, atmosphere, masks_wfs, object, build_dim=image_dim, noise=noise)
outfile = "$(folder)/Dr0_$(Dr0_composite)_ISH$(nsubaps_side)x$(nsubaps_side)_images$(id).fits"
writefits(observations_wfs.images, outfile, header=header)
outfile = "$(folder)/Dr0_$(Dr0_composite)_ISH$(nsubaps_side)x$(nsubaps_side)_psfs$(id).fits"
writefits(psfs_wfs, outfile, header=header)
###########################################
