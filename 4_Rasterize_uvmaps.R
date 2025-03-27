# Rasterize uv maps 
library(raster)

args <- commandArgs(trailingOnly = TRUE)
# Assign the arguments to variables
n_cores <- 4
if (length(args)) {
  species     <- args[1]
  LDS_SELECT  <- args[2]   
  pattern_col <- args[3] #"_CUR" # color to be processed 
  if (length(args) > 3) n_cores <- as.numeric(args[4]) 
} else{
  species     <- "cranes_souris"
  LDS_SELECT  <- "F_54" 
  pattern_col <- "_CUR"
}

# vector for luminance combination of rgb values
lum_vec <- c(0.299, 0.587, 0.114)

# parameters
#n_cores <- max(1, min(availableCores() - 1, n_cores))
raster_resolution <- 224  # User-defined resolution

# Paths 
#"C:/Users/ch1371gu/Desktop/POSTE BIOGEOSCIENCES/Projet Ecol+/MAPPING/"
path <- getwd()
path_initial_files <- paste0(path,"/data/",species,"AREAS_AROUND_LANDMARKS_", species, "/LDS_", LDS_SELECT,"/")
path2raster        <- paste0(path_initial_files, "RASTERS_tif")
path_param_files    <- paste0(path_initial_files, "param")
path_col_files    <- paste0(path_initial_files, "coloured")

# listing files
filenames_col_files <- list.files(path_col_files, pattern = pattern_col)
print(head(filenames_col_files))

filenames_param <- list.files(path_param_files, pattern = pattern_col)
filenames_param <- gsub("_lscm", "", filenames_param)

common_files <- intersect(basename(filenames_col_files), basename(filenames_param))

# Reorder obj files and param files
filenames_col_files_communs <- filenames_col_files[basename(filenames_col_files) %in% common_files]
filenames_param_communs <- filenames_param[basename(filenames_param) %in% common_files]

print(head(filenames_col_files_communs))
print(head(filenames_param_communs))

dir.create(file.path(path2raster))

#for (file_id in filenames_ldscm_communs){
rasterization <- function(file_id){
  # RASTERIZATION 
  print(file_id)
  output_file <- paste0(path2raster, "/raster_lscm_", gsub(".obj", "", file_id),".tif")
  
  if (file.exists(output_file)){
    print(paste("The file already exists: ", file_id))
    return(NULL) # next not supported in //
  }}
  
#   OBJ_initial_colore <- readLines(con  = paste0(path_col_files, file_id)) 
#   # extract the color info from the original file 
#   v <- OBJ_initial_colore[grep("v ", OBJ_initial_colore)];v <- gsub("v", "", v);v <- strsplit(v, " ");v <- as.numeric(unlist(v));v <- v[!is.na(v)]
#   # extract the RGB 3 columns matrix
#   rgb_mat <- matrix(v, ncol = 6, byrow = TRUE)[, 4:6]
#   # convert rbg values in luminance vector
#   luminance_tab <- rgb_mat %*% lum_vec
#   
#   # Extract texture information from uvmapped parameterization files 
#   OBJ_param_lscm <- readLines(con  = paste0(path_lscm_files, gsub(".obj", "", file_id), "_lscm.obj"))
#   vt <- OBJ_param_lscm[grep("vt", OBJ_param_lscm)]
#   vt <- gsub("vt", "", vt); vt <- strsplit(vt, " "); vt <- as.numeric(unlist(vt)) ; vt <- vt[!is.na(vt)]
#   vt <- matrix(vt, ncol = 2, byrow = TRUE) # get x-y coordinates
#   
#   # rasterisation LSCM
#   rastervide <- raster(ncols = raster_resolution, nrows = raster_resolution)
#   extent(rastervide) <- c(0,1,0,1)
#   crs(rastervide) <- ""
#   raster_LSCM <- rasterize(vt, rastervide, field = luminance_tab, fun = mean)
#   extent(raster_LSCM) <- c(0,1,0,1) 
#   names(raster_LSCM) <- file_id
# 
#   writeRaster(raster_LSCM, output_file, overwrite = TRUE)
#  # could add a tryCatch - error
# }
# 
# plan(multisession, workers = n_cores)
# furrr::future_map(filenames_ldscm_communs, rasterization)
# plan(sequential)

#library(progressr)
#handlers(global = TRUE)
#with_progress({
#  furrr::future_map(filenames_ldscm_communs, rasterization)
#})

