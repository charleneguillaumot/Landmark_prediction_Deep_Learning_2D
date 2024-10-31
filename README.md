# Landmark_prediction_Deep_Learning_2D
Use of Xception, ResNet101 and VIT Deep Learning model architectures to predict landmarks positioning on 3D mesh models using their parameterization in 2D subsets 

## Description of the project 
The developed pipeline starts with approximate landmark predictions obtained from the ALPACA global registration of a reference model on the object (Porto et al. 2021), a step which remains specific, and uses these predictions to subset the surface. The resulting local 3D surfaces are then parameterized in 2D images and colorized to enhance geometric features (ridges, flaws, hollows…) according to differential geometry and ambient lighting algorithms. The resulting raster images, quite generic to any bone structures, are associated with the manual landmark positions to train one Transformer and two Convolutional Neural Network (CNN) architectures. 

Input : 
- 3D mesh models (.ply files)
- landmarks
- landmark ALPACA predictions (see Step 0 below)

Output: 
-Deep Learning landmark prediction (x, y) position 

METTRE ICI L'IMAGE BILAN de la méthode 

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
The little mesh subsets needs to be colorized before being parameterized and converted into raster images. The Python code "3_Colorize_mesh_subsets.py" loads the 3D little images, clean them (closing holes, manifoldness correction) before colorizing them in 3 different channels (ambient occlusion, volumetric obscurance and APSS curvature). For one 3D little subset, 3 colorized little subsets are created and saved as .obj files. 

### Step 4. Parameterization of the colorized subsets 
The colorized subsets are transfered into the data/obj folder of the pmp-library architecture. The pmp library used in this project was modified from the one developed in https://www.pmp-library.org/ to enhance its calculation performances by removing several Viewer modules. The folder must be in a Linux environment (procédure pour lancer ? cmake, make et voilà?)
Expliquer le choix LSCM
Once the files to be parameterized transfered in the data/obj folder, access the build folder and do 
data_directory="../data/obj/"
for file in $(find "$data_directory" -name "*.obj"); do
	./parameterization "$file"
done
The parameterized images are then saved in the build/output_lscm directory 
