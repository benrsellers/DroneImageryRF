########################################################################

# Step 2: Random Forest Pixel Classification Model 
# Ben Sellers

# This Script uses ocular training data polygons of and creates a random forest 
# model using pixel values extracted from drone imagery. 
# The model is then projected across the entire imagery area to classify vegetation.
# Finally, the model is evaluated using validation data from using a user defined
# training/test split and error metrics are recorded. This script can be run multiple 
# times to test the influence of model parameters and training data on model outputs.

### Script Information ###
#1. Read in all shapefile data, assign classes, and merge into one shapefile.
#2. Split training polygons into calibration/validation sets and extract pixels from drone data for both sets
#3. Train an RF Model
#4. Run RF Model on the entire Camblin Ranch drone data stack
#5. Calculate error metrics from classification maps

# General scripts steps include:
# 1. Read in all ocular training data, assign classes to each for classification,
#    and merge them into a single shapefile. 

# 2. Split training polygons into a calibration and validation set. The worlflow currently
#    runs the split on the number of polygons given for each class. 

# 3. Train the RF Model by supplying it X data and Y data. This step also returns 
#    the out of bag error from the model, correlation analysis, and an importance plot for each band. 

# 4. Run the Random Forest Model on the raster stack provided to classify pixels and output a map.

# 5. Calculate error metrics (overall accuracy and Kappa) using validation data and export a confusion matrix.

# 6. Write all model inputs and metrics to ModelRuns.csv so to keep track of model runs and performance.

# INPUTS
# 1: Number of classes to use for classification
# 2: Names of classes used in classification
# 3: Path to drone_stack raster, created in Drone_RasterStack.R (drone_stack_path)
# 4: Path to the folder where model outputs are stored - script creates folder if it doesn't already exist (classification_out_path)
# 5: Random forest parameter mtry - determines how many variables to try at once in creating model (num_mtry) 
# 6: Random forest parameter trees - determines the number of trees used in creating model (num_trees)
# 7: Parameter to determine the proportion of training vs test data. EX: 0.8 = 80% training/test split (cal_val_split)
# OUTPUTS
# 1. ModelRuns folder containing subfolders for each random forest model
# 2. ModelRuns.csv containing spreadsheet of each model run's parameters and performance
# 3. ConfusionMatrix_***.png containing the confusion matrix from validation data compairson
# 4. "shp/calibration.shp" shapefile containing polygons used for model training
# 5. "shp/validation.shp" shapefile containing polygons used for model testing


########################################################################

# This script was built using R 4.4.1 

### PACKAGES ###


# Install necessary packages
# install.packages("terra")
# install.packages("sf")
# install.packages("randomForest")
# install.packages("caret")
# install.packages("dplyr")

# Load libraries
library(terra)
library(sf)
library(dplyr)
library(randomForest)
library(ggplot2)
library(tidyr)
library(caret)

# ----------------------------------------------------------------------
### Start Fill in Data ###
class_count = 7 # Number of classes 
class_names = c("EmergentVeg", "LightBrownVeg", "LightGreenVeg", "Phrag", "Rice", "Shrubs", "Water") #Names of classes
training_path <- file.path("E:/CoveringGround/repos/PB_data/overview_rgb/shp/PB_training") #Path to training data folder
drone_stack_path <- file.path("E:/CoveringGround/repos/PB_data/overview_rgb/tif/DroneStack/drone_stack.tif")
classification_out_path <- file.path("E:/CoveringGround/repos/PB_data/overview_rgb/tif/ModelOutputs")
num_mtry <- 7
num_trees <- 50
cal_val_split <- .8
### End Fill in Data ###
# ----------------------------------------------------------------------


###########################################################
### Step 0: SETUP setting file paths and output folders ###         
###########################################################

# Create folder for classification outputs, unless it already exists
if (!dir.exists(classification_out_path)) {
  dir.create(classification_out_path, recursive = TRUE)
  cat("Folder created successfully at", classification_out_path)
} else {
  cat("Folder already exists at", classification_out_path)
}

# Variable for todays dat to label RF outputs
current_date <- format(Sys.Date(), "%Y-%m-%d")

# Setting up variables so that we can run models in a loop and keep track of their run number
folder_count <- length(list.dirs(classification_out_path, recursive = F))+1

#creating folder for run
output_foldername <- file.path(classification_out_path, paste0("Run_", folder_count,'_', current_date))
dir.create(output_foldername, recursive = TRUE)

#create folder for output training data
calval_shp_path_out <- file.path(output_foldername,"shp")
dir.create(calval_shp_path_out, recursive = TRUE)

# spreadsheet for logging model runs and corresponding parameters
modelruns_path <- file.path(classification_out_path, "ModelRuns.csv")

# make new spreadsheet for ModelRuns unless it already exists
if (!file.exists(modelruns_path)) {
  # Create a new CSV with sample data (or an empty one)
  data <- data.frame(Run_num = character(), mtry = numeric(), trees = numeric(), calval_split = numeric(),
                     Shrub_pix = numeric(), Bare_pix = numeric(), Grass_pix = numeric(), OverallAccuracy = numeric(), Kappa = numeric(), OOB = numeric())
  write.csv(data, modelruns_path, row.names = FALSE)
  
  cat("File created:", modelruns_path, "\n")
} else {
  cat("File already exists:", modelruns_path, "\n")
}


#Test
# Make new spreadsheet for ModelRuns unless it already exists
if (!file.exists(modelruns_path)) {
  # Dynamically create column names for each class's pixel count based on class names
  class_columns <- paste0(class_names, "_pix")
  
  # Create a new CSV with sample data (or an empty one) including pixel counts for each class
  data <- data.frame(
    Run_num = character(),
    mtry = numeric(),
    trees = numeric(),
    calval_split = numeric(),
    OverallAccuracy = numeric(),
    Kappa = numeric(),
    OOB = numeric(),
    # Dynamically add the class columns based on class_names
    matrix(ncol = length(class_names), nrow = 0)  # Empty matrix for class pixel columns
  )
  
  # Convert matrix into appropriate column names
  colnames(data)[8:(7 + length(class_names))] <- class_columns
  
  # Write the empty data frame to CSV
  write.csv(data, modelruns_path, row.names = FALSE)
  
  cat("File created:", modelruns_path, "\n")
} else {
  cat("File already exists:", modelruns_path, "\n")
}


#
# Generate dynamic paths for each class
training_paths <- lapply(1:class_count, function(i) {
  file.path(paste0(training_path, "/", i, "_", class_names[i],"TrainingData"))
})
training_paths
################################################
### Step 1. Merging Training Data Shapefiles ###         
################################################


# Function for reading in all training polygon shapefiles and joining them into one sf polygon
read_and_merge_training_polygons <- function(filepaths) {
  #make an empty sf object to store joined shapefiles
  shapefiles <- NULL
  
  #loop over shapefiles in the list of filepaths you supply
  for (i in filepaths) {
    shp = st_read(i)
    if (is.null(shapefiles)) {
      shapefiles = shp
    } else {
      shapefiles <- rbind(shapefiles, shp)
    }
  } 
  return(shapefiles)
}

read_and_merge_training_polygons <- function(filepaths, class_id) {
  shapefiles <- do.call(rbind, lapply(filepaths, function(path) {
    shp <- st_read(path)
    shp$class <- class_id
    shp
  }))
  return(shapefiles)
}

# Combine shapefiles for all classes
training_poly_list <- lapply(1:class_count, function(i) {
  filepaths <- list.files(training_paths[[i]], pattern = ".shp$", full.names = TRUE)
  read_and_merge_training_polygons(filepaths, i)
})
training_poly_list

# Merge all classes into one sf object
training_poly <- do.call(rbind, training_poly_list)

# Add row index for identification
training_poly <- training_poly %>% mutate(index = rownames(.))

# Validate geometries
training_poly <- training_poly[st_is_valid(training_poly), ]

#########################################
### Step 2. Training/Validation Split ###
#########################################

# read dronestack raster
drone_stack <- rast(drone_stack_path)

# Select polygons to use for calibration, splits each class by cal/val split
calibration_set <- training_poly %>%
  group_by(class) %>%
  slice_sample(prop = cal_val_split, replace = FALSE) %>%
  ungroup() %>% 
  mutate(ID = seq_len(nrow(.)))
st_write(training_poly, file.path(calval_shp_path_out, "calibration.shp"), append = FALSE)

# Select the remaining validation (test data)
validation_set <- training_poly %>%
  filter(!st_equals(., calibration_set, sparse = FALSE) %>% rowSums()) %>% 
  mutate(ID = seq_len(nrow(.)))
st_write(training_poly, file.path(calval_shp_path_out, "validation.shp"), append = FALSE)

#extract pixels from calibration polygon subset and join them to the polygon that they were extracted from
extracted_cal_values <- terra::extract(drone_stack, calibration_set, xy= TRUE)#, bind = TRUE)#, sp = TRUE)# %>% na.omit(extracted_cal_values)
extracted_cal_values <- extracted_cal_values  %>% left_join(calibration_set, by = "ID", select(ID))

#get the number of pixels in each class to report in ModelRuns.csv
pixel_counts <- sapply(1:class_count, function(i) {
  sum(extracted_cal_values$class == i, na.rm = TRUE)
})

#extract pixels from validation polygon subset
extracted_val_values <- terra::extract(drone_stack, validation_set, xy= TRUE)# %>% na.omit(extracted_val_values)
extracted_val_values <- extracted_val_values  %>% left_join(validation_set, by = "ID", select(ID))
table(extracted_val_values$class)

####################################
### Step 3. Creation of RF Model ###
####################################

#Making the class column a factor
extracted_cal_values$class <- factor(extracted_cal_values$class)
extracted_val_values$class <- factor(extracted_val_values$class)

# Define predictor variables for modeling
x_data <- extracted_cal_values[1:15]
x_data <- subset(x_data, select= -c(ID,alpha,x,y))
x_data
#define dependent variable for modeling
y_data <- extracted_cal_values$class
y_data <- as.factor(y_data)

y_data

# Line to show corellation between input data
cor(x_data)

#set seed to be reproducible 
set.seed(0)

#train model
rf_classifier <- randomForest(x=x_data, y=y_data, mtry=num_mtry, type = 'response', ntree=num_trees, importance=TRUE, keep.forest=TRUE)

#view the model
rf_classifier
plot(rf_classifier)

#get the out of bag error from the model, to add to ModelRuns.csv
OOB_error <- rf_classifier$err.rate[nrow(rf_classifier$err.rate), "OOB"]
OOB_error

#line to plot the importance of each variable 
importance(rf_classifier)
varImpPlot(rf_classifier, main = 'Band Importance')

#################################################################
### Step 4. Classifying Camblin Ranch Drone Map with RF Model ###
#################################################################

#creating file name for output tif
output_filename <- file.path(output_foldername, paste0("Run",folder_count,'_', current_date, ".tif"))
output_filename

drone_stack[is.na(drone_stack)] <- 0

# Running RF on the drone stack
map <- predict(drone_stack, type='response', rf_classifier, filename=output_filename, format="GTiff", overwrite=TRUE)
plot(map)
############################################################
### Step 5. Accuracy assement using ocular training data ###
############################################################

# Defining Calibration and Validation shapefiles
validation <- vect(extracted_val_values[, c("class", "x", "y")],geom = c("x", "y"))
calibration <- vect(extracted_cal_values[, c("class", "x", "y")], geom = c("x", "y"))

# Extracting pixel values from the classified map using the training sample validation points
validation$reference <- as.factor(validation$class)
validation$reference
prediction <- terra::extract(map, validation)
prediction
# adding the predicted pixel values to the df with the known pixel values (from training polygons)
validation$prediction <- as.factor(prediction$class)

# Creating a confusion matrix using the caret package
cm <- confusionMatrix(as.factor(validation$prediction), as.factor(validation$reference))
print(cm)

# Getting model metrics to be reported in ModelRuns.csv
OA <- cm$overall["Accuracy"]
kappa <- cm$overall["Kappa"]

# making output from caret confusion matrix able ot be plotted
hm <- as.data.frame(as.table(cm))
hm
hm$Prediction <- factor(hm$Prediction, levels = 1:class_count, labels = class_names)
hm$Reference <- factor(hm$Reference, levels = 1:class_count, labels = class_names)

# Dynamic class labeling for plot
plot <- ggplot(hm, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  theme_bw() +
  coord_equal() +
  scale_fill_distiller(palette = "Greens", direction = 1) +
  guides(fill = "none") +
  geom_text(aes(label = Freq), color = "black", size = 10) +
  scale_x_discrete(limits = class_names, labels = class_names) +
  scale_y_discrete(limits = class_names, labels = class_names) +
  # Text size adjustments (optional)
  theme(
    axis.text.x = element_text(size = 20, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 20),
    axis.title.x = element_text(size = 30),
    axis.title.y = element_text(size = 30)
  )

# Display the plot
plot
   
#save plot to output folder
ggsave(file.path(output_foldername, paste0("ConfusionMatrix_", folder_count, '_', current_date, ".png")), width = 15, height = 8)


########################################################
### Step 5: Write all model metrics to ModelRuns.csv ###
########################################################

# Create the data frame row for this model run
new_row <- data.frame(
  Run_num = folder_count,  # Assuming `folder_count` is defined earlier
  mtry = num_mtry,         # Assuming `num_mtry` is defined earlier
  trees = num_trees,       # Assuming `num_trees` is defined earlier
  calval_split = cal_val_split,  # Assuming `cal_val_split` is defined earlier
  OverallAccuracy = OA,    # Assuming `OA` is defined earlier
  Kappa = kappa,           # Assuming `kappa` is defined earlier
  OOB = OOB_error          # Assuming `OOB_error` is defined earlier
)

# Add pixel counts for each class dynamically to the row
for (i in 1:class_count) {
  new_row[[paste0(class_names[i], "_pix")]] <- pixel_counts[i]
}
new_row

# Write Row to table
write.table(new_row, modelruns_path, row.names = FALSE, col.names = FALSE, sep = ",", append = TRUE)


