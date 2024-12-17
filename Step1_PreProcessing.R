########################################################################

# Step 2: Raster Calculations and Mosaicing 
# Ben Sellers

# Step 1 uses RGB and DSM/DTM layers from processed drone imagery 
# to calculate vegetation indices and terrain metrics and creates a composite for 
# use in supervised classification. 

# General scripts steps include:
#1. Read in all raster data, smooth DSM and DTM to remove NAs and resample RGB 
#   orthomosaic to have same resolution as DSM/DTM.

#2. Calculate RGB vegetation indices: VDVI, GRRI, NGRDI, and VARI. 

#3. Calculate structural metrics: Canopy height model, Roughness, TPI, and TRI.

#4. Rename raster bands and stack them into a composite raster, write the output to output folder


# INPUTS
# 1: Path to the rgb orthomosaic raster. Band 1 = Red, Band 2 = Green, Band 3 = Blue, Band 4 = Alpha)(rgb_raster_path)
# 2: Path to digital terrain model raster (dtm_raster_path)
# 3: Path to digital surface model raster (dsm_raster_path)
# 4: Path where raster calculations and final raster stack should be written to (output_folder_path)

# OUTPUTS
# 1. DTM with extrapolated NA values
# 2. DSM with extrapolated NA Values
# 3. Resampled RGB orthomosaic
# 4. Individual RGB vegetation index rasters
# 5. Terrain raster with bands for TRI, TPI, and Roughness
# 6. Composite raster with 14 bands


########################################################################

# Install necessary packages
# install.packages("terra")

# Load libraries
library(terra)

# ----------------------------------------------------------------------
### Start Fill in Data ###
rgb_raster_path <- file.path("E:/CoveringGround/repos/PB_data/overview_rgb/tif/PheasantBranch_RGB3cm.tif") # Required
dtm_raster_path <-NA # Optional. Leave this as N/A if you dont have a dtm
dsm_raster_path <- file.path("E:/CoveringGround/repos/PB_data/overview_rgb/tif/PheasantBranch_DEM.tif") # Required
output_folder_path <- file.path("E:/CoveringGround/repos/PB_data/overview_rgb/tif/DroneStack") # Required
### End Fill in Data ###
# ----------------------------------------------------------------------

###################################################
### Step 0: Setup file paths and output folders ###         
###################################################

# Create the outputs folder
if (!dir.exists(output_folder_path)) {
  dir.create(output_folder_path, recursive = TRUE)
  cat("Folder created successfully at", output_folder_path)
} else {
  cat("Folder already exists at", output_folder_path)
}

##########################################################
###   Step 1: Fill NAs and Resample RGB orthomosaic    ###
##########################################################

#Reading in Drone layers 
if (!is.na(dtm_raster_path)) {
  drone_dtm <- rast(dtm_raster_path)
} else {
  print("No DTM Raster Provided")
}
drone_dsm <- rast(dsm_raster_path) #"data/tif/drone_dsm_focal.tif") #This layer was made in Moffat_Random_Forest_Sellers to remove NAs using focal smoothing
drone_rgb <- rast(rgb_raster_path)#This layer was mad3 in Moffat_Random_Forest_Sellers to resample RGB to the resolution of DSM and DTM (10cm)

# Resample RGB so pixels are same size as DSM/DTM
drone_rgb <- resample(drone_rgb, drone_dsm,threads=TRUE)
writeRaster(drone_rgb, file.path(output_folder_path,"drone_rgb10cm.tif"), overwrite = TRUE)

# Fill NA cells in DTM/DSM rasters
drone_dsm <- focal(drone_dsm, w=9, fun=mean, na.policy="only", na.rm=T)
writeRaster(drone_dsm, file.path(output_folder_path,"drone_dsm_focal.tif"), overwrite = T)
if (!is.na(dtm_raster_path)) {
  drone_dtm <- focal(drone_dtm, w=9, fun=mean, na.policy="only", na.rm=T)
  writeRaster(drone_dtm, file.path(output_folder_path,"drone_dtm_focal.tif"), overwrite = T)
} else {
  print("No DTM Raster Provided")
}

# Rename Bands
names(drone_rgb) <- c("red","green", "blue", "alpha")
if (!is.na(dtm_raster_path)) {
  names(drone_dtm) <- "dtm"
} else {
  print("No DTM Raster Provided")
}
names(drone_dsm) <- "dsm"
###########################################
###       Step 2: Calculate RGB VIs     ###
###########################################


# Calculating RGB Vegetation Indices
VDVI = (2*drone_rgb$green-drone_rgb$red - drone_rgb$blue)/(2*drone_rgb$green+drone_rgb$red+drone_rgb$blue)
writeRaster(VDVI, file.path(output_folder_path,"VDVI.tif"), overwrite = T)
NGRDI = (drone_rgb$green-drone_rgb$red)/(drone_rgb$green+drone_rgb$red)
writeRaster(NGRDI, file.path(output_folder_path,"NGRDI.tif"), overwrite = TRUE)
VARI = (drone_rgb$green-drone_rgb$red)/(drone_rgb$green+drone_rgb$red-drone_rgb$blue)
writeRaster(VARI, file.path(output_folder_path,"VARI.tif"), overwrite = TRUE)
GRRI = drone_rgb$green/drone_rgb$red
writeRaster(GRRI, file.path(output_folder_path,"GRRI.tif"), overwrite = TRUE)

###########################################
### Step 2: Calculate Structural Indices###
###########################################

#Create canopy height model by subtracting DTM from DSM
if (!is.na(dtm_raster_path)) {
  CHM <- drone_dsm - drone_dtm
  writeRaster(CHM, file.path(output_folder_path,"drone_CHM.tif"), overwrite = TRUE)
} else {
  print("No DTM Raster Provided, can't calc CHM")
}

# Using terra's built in terrain characteristics function to get TPI, TRI, and Roughness from drone DSM
terrain_metrics <- terrain(drone_dsm, v=c("TRI", "TPI", "roughness"), neighbors = 8, filename=file.path(output_folder_path,"terrain.tif"))

###########################################
###     Step 3: Stack  Rasters          ###
###########################################

# Stack all of the layers together

if (!is.na(dtm_raster_path)) {
  drone_stack <- c(drone_rgb, drone_dsm, drone_dtm, CHM, terrain_metrics$TRI, terrain_metrics$TPI,
                   terrain_metrics$roughness, VDVI, NGRDI, VARI, GRRI)
} else {
  drone_stack <- c(drone_rgb, drone_dsm, terrain_metrics$TRI, terrain_metrics$TPI,
                   terrain_metrics$roughness, VDVI, NGRDI, VARI, GRRI)
  
}


#rename the drone layer names
if (!is.na(dtm_raster_path)) {
  names(drone_stack) <- c("r", "g", "b", "alpha", "dsm", "dtm", "chm", "tri", "tpi",
                          "roughness", "vdvi", "ngrdi", "vari", "grri")
} else {
  names(drone_stack) <- c("r", "g", "b", "alpha", "dsm", "tri", "tpi",
                          "roughness", "vdvi", "ngrdi", "vari", "grri")
  
}

#changing all no data values to 0 so that RF can run on raster in next step
drone_stack[is.na(drone_stack)] <- 0

# Write Drone stack raster
writeRaster(drone_stack, file.path(output_folder_path,"drone_stack.tif"), overwrite = TRUE)
