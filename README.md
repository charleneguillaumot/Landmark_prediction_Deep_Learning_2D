# Landmark_prediction_Deep_Learning_2D
Use of Xception, ResNet101 and VIT Deep Learning model architectures to predict landmarks positioning on 3D mesh models using their parameterization in 2D subsets 

## Description of the project 
xxxx détailler 
Input : 
- 3D mesh models (.ply files)
- landmarks
- landmark ALPACA predictions (see Step 0 below)

Output: 
-Deep Learning landmark prediction

xxx Pluggin Slicer ????


## Pipeline 
### Folder arborescence 
Détailler ici les dossiers et leurs contenus 

### Step 0. Generate ALPACA's landmark predictions 
The aim is to have, for each individual, several sets of landmarks predictions (ALPACA) that will play the role of barycenters in Step 2, to extract small portions of the 3D meshes in the area of interest of a landmark 
You can generate the landmark predictions using the software Slicer and the SlicerMorph pluggin manually if you want, 
otherwise, we developped a script for running ALPACA in batch, with varying alpha and beta parameters (to have several ALPACA predictions replicates efficiently)
Please see the code and procedure at https://github.com/charleneguillaumot/ALPACA_from_terminal
In our code, we used 4 ALPACA's predictions.

### Step 1. Rework initial ply files 
In the next step (Step 2), we will need to have proper .ply files to be able to split the inside and outside layers of the mesh. For that, coloration by ambient occlusion is necessary. 
The "1_Rework_initial_ply_files.py" file simply loads your .ply files, colorizes them by an ambient occlusion filter and save a copy in your folder.

### Step 2. Mesh subsetting of little 3D landmark zones
The code "2_Mesh_subsetting_landmark_zones.R" (written in R, as it uses a R library developped in our lab), splits the inside and outside layers of the 3D mesh and then extracts a little 3D zone barycentered by an ALPACA prediction. 
This enables to have as outputs little 3D objects, where we will perform the landmark position research by the Deep Learning model. 

### Step 3. Mesh colorization into 3 color channels : ambient occlusion, volumetric obscurance and curvature 
The little mesh subsets needs to be colored before being parameterized and converted into raster images 
