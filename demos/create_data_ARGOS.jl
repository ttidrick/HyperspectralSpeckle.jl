include("../src/mfbd.jl")
using Main.MFBD

############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "data/test"
id = ""
verb = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64

nsubaps_side = 6
nepochs = 100
nλ = 101
λ_nyquist = 400.0
λ_ref = 500.0
λmin = 400.0
λmax = 1000.0
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
masks_wfs = Masks(
    dim=image_dim,
    nsubaps_side=nsubaps_side, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
masks = [masks_full, masks_wfs]
# masks = [masks_full]
[writefits(masks[dd].masks, "$(folder)/masks_ISH$(masks[dd].nsubaps_side)x$(masks[dd].nsubaps_side)$(id).fits", header=header) for dd=1:length(masks)]
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
fov = 30.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full .* nsubaps_side
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
rn = 2.0
exptime = 5e-3
noise = true
ζ = 0.0
########## Create Optical System ##########
oap_full = repeat([OpticalElement(name="Thorlabs:OAP-P01", FTYPE=FTYPE)], 2)
mirrors_full = repeat([OpticalElement(name="Thorlabs:Plano-P01", FTYPE=FTYPE)], 2)
dichroic_full = OpticalElement(name="Thorlabs:DMLP805P-transmitted", FTYPE=FTYPE)
lens_full = OpticalElement(name="Thorlabs:AB", FTYPE=FTYPE)
optics_full = OpticalSystem(cat(oap_full..., mirrors_full..., dichroic_full, lens_full, dims=1), λ, verb=verb, FTYPE=FTYPE)
######### Create Full-Ap Detector #########
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
observations_full = Observations(
    optics_full,
    detector_full,
    ζ=ζ,
    D=D,
    nepochs=nepochs,
    nsubaps=1,
    dim=image_dim,
    verb=verb,
    FTYPE=FTYPE
)

oap_wfs = OpticalElement(name="Thorlabs:OAP-P01", FTYPE=FTYPE)
mirrors_wfs = repeat([OpticalElement(name="Thorlabs:Plano-P01", FTYPE=FTYPE)], 2)
dichroic_wfs = OpticalElement(name="Thorlabs:DMLP805P-reflected", FTYPE=FTYPE)
lens_wfs = repeat([OpticalElement(name="Thorlabs:AB", FTYPE=FTYPE)], 3)
mla_wfs = OpticalElement(name="Thorlabs:MLA-AR", FTYPE=FTYPE)
optics_wfs = OpticalSystem(cat([oap_wfs, mirrors_wfs..., dichroic_wfs, lens_wfs..., mla_wfs], dims=1), λ, verb=verb, FTYPE=FTYPE)
detector_wfs = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_wfs,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
observations_wfs = Observations(
    optics_wfs,
    detector_wfs,
    ζ=ζ,
    D=D,
    nepochs=nepochs,
    nsubaps=masks_wfs.nsubaps,
    dim=wfs_dim,
    verb=verb,
    FTYPE=FTYPE
)
observations = [observations_full, observations_wfs]
# observations = [observations_full]
###########################################

############ Object Parameters ############
objectfile = "data/OCNR2.fits"
~, spectrum = solar_spectrum(λ=λ)
template = false
mag = 4.0
background_mag = FTYPE(Inf)
broadband_filter = OpticalElement(λ=λ, response=ones(FTYPE, nλ), FTYPE=FTYPE)
flux = mag2flux(λ, spectrum, mag, broadband_filter, D=D, ζ=ζ, exptime=exptime)
background_flux = mag2flux(λ, ones(nλ), background_mag, broadband_filter, D=D, ζ=ζ, exptime=exptime)
object_height = 515.0  # km
############## Create object ##############
object = Object(
    flux=flux,
    background_flux=background_flux,
    λ=λ,
    fov=fov,
    height=object_height, 
    dim=image_dim,
    spectrum=spectrum,
    objectfile=objectfile, 
    template=template,
    verb=verb,
    FTYPE=FTYPE
)
writefits(object.object, "$(folder)/object_truth$(id).fits", header=header)
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
Dr0_vertical = 20.0
Dr0_composite = Dr0_vertical * sec(ζ*pi/180)
r0_composite = D / Dr0_composite
heights = [0.0, 7.0, 12.5]
wind_speed = wind_profile_roberts2011(heights, ζ)
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
calculate_layer_masks_eff!(patches, atmosphere, observations_full, object, masks_full)
create_phase_screens!(atmosphere, observations_full)
# opd_smooth = calculate_smoothed_opd(atmosphere, observations_full)
calculate_composite_pupil_eff(patches, atmosphere, observations, object, masks, build_dim=image_dim, propagate=propagate)
atmosphere.opd .*= atmosphere.masks[:, :, :, 1]
# atmosphere.ϕ .*= atmosphere.masks

# writefits(atmosphere.masks, "$(folder)/layer_masks.fits", header=header)
# writefits(observations_full.A, "$(folder)/Dr0_$(Dr0_composite)_ISH1x1_amplitude$(id).fits")
# writefits(observations_wfs.A, "$(folder)/Dr0_$(Dr0_composite)_ISH$(nsubaps_side)x$(nsubaps_side)_amplitude$(id).fits")
# writefits(atmosphere.ϕ, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_phase_full$(id).fits", header=header)
# writefits(patches.ϕ_slices, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_phase_slices$(id).fits", header=header)
# writefits(patches.ϕ_composite, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_phase_composite$(id).fits", header=header)
writefits(atmosphere.opd, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full$(id).fits")
# writefits(opd_smooth, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_opd_full_smooth$(id).fits")
###########################################

########## Create Full-Ap images ##########
create_images_eff(patches, observations, atmosphere, masks, object, build_dim=image_dim, noise=noise)
[writefits(observations[dd].images, "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_images$(id).fits", header=header) for dd=1:length(observations)]
# outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH1x1_monochromatic_images$(id).fits"
# writefits(observations_full.monochromatic_images, outfile, header=header)
# outfile = "$(folder)/Dr0_$(round(Int64, Dr0_composite))_ISH1x1_psfs$(id).fits"
# writefits(patches.psfs[1], outfile, header=header)
###########################################
