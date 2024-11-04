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
The arborescence is the following 
* DATAS_prep
	1. INDIVIDUALS
		* Mandible_primates
			1. landmarks_folder: contains the list of .fscv files detailling the (x,y,z) position of the ground truth images
			2. surfaces_folder: containts the 3D surface .ply files (surface mesh models)
			3. [output_alpaca]: folder generated at step 0
	2. AREAS_AROUND_LANDMARKS
		* [LDS_nb]: for each studied landmark, an equivalent folder will be generated
			1. [colorless] : folder created at step 2, contains the subsets of the whole 3D surface initial files in the area of interest (ALPACA's predictions used as barycenters)
			2. [RAW_uvmapping/output_lscm]: folder generated at step 4 which contains the parameterized .obj files
			3. [RASTERS_tif]: folder generated at step 5 which contains the rasters .tif files, which are sorted in 3 subfolders (AO, VO, CUR) at step 6
			4. [LDS_uv_coords]: folder created during step 7 which contains .txt files with the coordinates of the ground truth landmark positions in the 2D space

The steps to generate the training images are the following: 
 
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


### Step 5. Rasterization of the parameterized images
The .obj files are then opened in R, and rasterized into a flat 2D images of 224x224 pixel dimension (extent 0.1x0.1), colorized by the color texture saved in the .obj file and saved as a .tif file for the next steps 

### Step 6. Cleaning and sorting of raster tif files 
The tif files are loaded and analysed to spot images that are mostly black (empty pixels, the ones for which the parameterization did not work). The threshold was set up at 99% of black pixels. Some trials were performed with a variation at 95%, showing no significant modifications in model performances final results. The files are then sorted in  different folders according to their color (AO : ambient occlusion, VO: volumetric obscurance, CUR: APSS curvature).

### Step 7. Transfer the 3D ground truth landmark positions on the 2D space 
Using the KDtree approach from the scipy library, the last step is to prepare the list of ground truth coordinates in the 2D space for model training along with the raster images. 

