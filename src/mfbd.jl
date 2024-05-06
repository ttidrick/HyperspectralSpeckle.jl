module MFBD

## masks
include("masks.jl")
export Masks, make_ish_masks, make_simple_mask

## imaging
include("imaging.jl")
export create_refraction_operator, create_extractor_operator, create_extractor_adjoint, pupil2psf, pupil2psf!, position2phase, position2phase!, add_noise!
export create_monochromatic_image, create_monochromatic_image!, create_polychromatic_image, create_polychromatic_image!

## isoplanatic
include("isoplanatic.jl")
export calculate_layer_masks_iso!, calculate_composite_pupil_isoplanatic!, create_images_isoplanatic!, calculate_smoothed_opd

include("anisoplanatic.jl")
export AnisoplanaticPatches, get_center, get_center!, create_patch_extractors, create_patch_extractors_adjoint, calculate_layer_masks_eff!, calculate_layer_masks_eff_alt!
export calculate_composite_pupil_eff, calculate_composite_phase_eff!, calculate_composite_amplitude_eff!, create_images_eff, create_images_eff!, change_heights!

## observations
include("observations.jl")
export Filter, Detector, Observations

## object
include("object.jl")
export Object, mag2flux, template2object, interpolate_object, poly2object, poly2object!, object2poly, object2poly!

## atmosphere
include("atmosphere.jl")
export Atmosphere, create_phase_screens!, calculate_screen_size!, calculate_pupil_positions!
export get_refraction, propagate_layers, wind_profile_greenwood, wind_profile_roberts2011, interpolate_phase

## criterion
include("criterion.jl")
export loglikelihood_gaussian
export fg_object, fg_phase, fg_opd

## regularization
include("regularization.jl")
export Regularizers, no_reg, tv2_reg, l2_reg, λtv_reg

## deconvolution
# include("deconvolution.jl")
# export Reconstruction, Helpers, reconstruct_blind!, reconstruct_myopic!

## deblurring
# include("deblurring.jl")
# export PatchHelpers, reconstruct_blind_aniso!, height_solve!

## reconstruct
include("reconstruct.jl")
export Reconstruction, Helpers, PatchHelpers, reconstruct_blind!, object_solve!, opd_solve!, height_solve!

## plot
include("plot.jl")
export plot_object, plot_recovered_layers

## utils
include("utils.jl")
export gettype, writefits, writefile, writeobject, readobject, readfile, readqe, readimages, readmasks, readfits, read_spectrum, vega_spectrum, solar_spectrum
export gaussian_kernel, calculate_entropy, shift_and_add, fit_plane, crop, smooth_to_rmse!, bartlett_hann2d, super_gaussian, setup_ifft, setup_operator_mul, block_reduce!, block_reduce, block_replicate!, block_replicate

## TOP SECRET!
include("government_secrets.jl")
export show_the_sausage, show_the_satellite

end