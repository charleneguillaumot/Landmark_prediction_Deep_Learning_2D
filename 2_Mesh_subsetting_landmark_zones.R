# Usage:
# Rscript 2_Mesh_subsetting_landmark_zones.R species2process landmark_2_process n_cores2use
# Rscript 2_Mesh_subsetting_landmark_zones.R  "primates" "F-2" 4

library(Morpho)        # read.fcsv()
library(Rvcg)
library(furrr)

source("utils_mesh.R") # local copies of digit3DLand functions

args <- commandArgs(trailingOnly = TRUE)
# Assign the arguments to variables
n_cores <- 4
if (length(args)) {
  species <- args[1]
  species <- "souris"
  LDS     <- args[2]   
  if (length(args) > 2) n_cores <- as.numeric(args[3]) 
} else{
  LDS     <- "F_15" 
  species <- "cranes_souris"
}

# parameters
n_cores      <- max(1, min(availableCores() - 1, n_cores))
change       <- "geodesic"
verbose      <- TRUE
zoomPercDist <- c(0.04,0.05,0.06)
translation  <- c(1, 0.95)
bone <- c("Mandible", "Skull")[2]

# "C:/Users/ch1371gu/Desktop/POSTE BIOGEOSCIENCES/Projet Ecol+/MAPPING/"
working_path <- getwd()
mainDir <- paste0(working_path,"/data/",species,"/AREAS_AROUND_LANDMARKS_", species,"/")

# CREATE EMPTY FOLDERS THAT WILL RECEIVE THE GENERATED SUBSETS 
#----------------------------------------------------------
subDir <- paste0("LDS_", LDS)
dir.create(file.path(mainDir), showWarnings = FALSE)
dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
dir.create(file.path(paste0(mainDir,subDir,"/"), "colorless/"), showWarnings = FALSE)

## LOAD THE ALPACA PREDICTED LANDMARKS FOLDER
dir_fichier_lds  <- paste0(working_path, "/data/", species, "/output_rigid/")
DIRnames_lds_dir <- list.files(dir_fichier_lds, full.names = TRUE)
if (verbose) print(basename(DIRnames_lds_dir))

# read all mesh files once...
folder_skulls <- paste0(working_path,"/data/", species, "/", bone, "_Surfaces_lisses/")
filenames_folder_skulls <- list.files(folder_skulls, full.names = TRUE)

# LOAD THE LANDMARKS
#--------------------
# Loop per individual -> future_map
for (DIR in DIRnames_lds_dir[1]){
  subset_mesh_list <- function(DIR){  
  INDI_SELECT <- basename(DIR) #-> basename was not in the original code, replace by DIR but to check ?
  filename_lds <- list.files(DIR)
  if (verbose) {
    print(paste0("Processing individual: ", DIR))
    print(paste0("List of fcsv files, containing ALPACA's landmark predictions for the individual", basename(DIR)))
    print(head(filename_lds))
  }
  # REMOVE THE INTERNAL MESH LAYER
  #--------------------------------
  # instead of vcgPlyRead default we could directly use vcgImport and fix update normals and clean args
  # as we probably don't need to clean anymore 
  skull_ambient_occ <- vcgImport(paste0(folder_skulls,  paste0(basename(DIR),"_AO.ply")), 
                                 updateNormals = FALSE, clean = FALSE, readcolor = TRUE)
  skull_ambient_occ$material <- NULL
  
  # spot the threshold for internal and external layers for non watertight meshes using ambient occlusion luminosity 
  # hist(skull_ambient_occ$quality, main=paste0("Indi",INDI_SELECT))
  
  # Get external mesh based on AO
  skull <- subset.mesh3d(skull_ambient_occ, skull_ambient_occ$quality > 0.02)
  skull$material <- NULL

  # LOOP TO RUN THIS CODE FOR THE 4 ALPACAS PREDICTIONS
  for (select_alpaca_pred in 1:4){
    # Extract the coordinates of landmark LDS for the 4 ALPACA predictions 
    lds_pred_ALPACA <- read.fcsv(paste0(DIR, "/", filename_lds[select_alpaca_pred]))
    lds_pred_ALPACA_lds_interet <- lds_pred_ALPACA[LDS, ]
    if (verbose) print(lds_pred_ALPACA_lds_interet)
    Pt <- lds_pred_ALPACA_lds_interet
    kd <- vcgKDtree(target = skull, query = matrix(lds_pred_ALPACA[LDS, ], nrow = 1, ncol = 3), k = 1)

      for (zoom in zoomPercDist){
        for (trans in translation){
          tryCatch({
            if (change == "geodesic"){
              geo <- vcgDijkstra(skull, kd$index)
              to_keep <- geo < max(geo)*zoom
            } else {
              dd <- sqrt(colSums((skull$vb[-4,]-Pt*trans)^2)) # Calculates the distance between the initial point and all mesh vertices 
              maxRad <- zoom * max(dd)
              to_keep <- dd < maxRad
            }
            if (!any(to_keep)){
                 print(paste0("Error for individual", filename_lds[select_alpaca_pred],"zoom_",zoom,"trans_",trans))
            } 
            else {
            subset_skull <- subset.mesh3d(skull, subset = to_keep) # on créé un subset du mesh originel
              
            if (length(subset_skull$vb) > 500){
              IdxVert <- 1:nrow(skull$vb[-4, ])
              tmp <- digit3DLand_vcgIsolatedIndex(subset_skull)
              isol <- tmp[[1]]
              idx_vb <- tmp[[2]]
              vd <- lapply(isol, distMin, list(vb = matrix(c(Pt, 1), 4, 1)))
              vd <- matrix(unlist(vd), nrow = length(isol), ncol = 2, byrow = TRUE)
              idxL <- which.min(vd[, 1])
              subset_skull2 <- isol[[idxL]]
              
              filename <- paste0(file.path(mainDir, subDir),"/colorless/subset_",zoom*100,"perc_trans_",trans,"_lds_",LDS,"_rep_",select_alpaca_pred,"_indi_",INDI_SELECT,".ply")
              vcgPlyWrite(subset_skull2, filename = filename)

            } 
            else {
              print("SECOND TYPE OF ERROR: less than 500 vertices, wrong little mesh")
            }
          }
        }) # end try-catch
    } 
  }
}}}

plan(multisession, workers = n_cores)
furrr::future_map(DIRnames_lds_dir[1], subset_mesh_list)
#plan(sequential)
