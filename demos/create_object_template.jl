include("../src/mfbd.jl")
using Main.MFBD

############# Data Parameters #############
FTYPE = Float32;
folder = "/home/dan/Desktop/ApJL_2024/separations/recon/Delta_mag10"
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
fov = 20  # arcsec
flux_ratio = 10^(-4)

plate_scale = fov / image_dim  # arcsec / pix
θ_arcsec = 0.5  # arcsec
θ_pix = θ_arcsec / plate_scale
Δx = round(Int, θ_pix / sqrt(2))

object = zeros(FTYPE, image_dim, image_dim)
object[image_dim÷2+1, image_dim÷2+1] = 1.0
object[image_dim÷2+1-Δx, image_dim÷2+1+Δx] = flux_ratio
writefits(object, "$(folder)/binary_0.5arcsec_DeltaMag10.fits")
###########################################