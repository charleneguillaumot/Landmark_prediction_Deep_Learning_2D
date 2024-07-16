# Landmark_prediction_Deep_Learning_2D
Use of Xception, ResNet101 and VIT Deep Learning model architectures to predict landmarks positioning on 3D mesh models using their parameterization in 2D subsets 

## Détailler le projet ici
Input : 

## Pipeline 
### Folder arborescence 
Détailler ici les dossiers et leurs contenus 

### Step 0. Generate ALPACA's landmark predictions 
The aim is to have, for each individual, several sets of landmarks predictions (ALPACA) that will play the role of barycenters in Step 2, to extract small portions of the 3D meshes in the area of interest of a landmark 
You can generate the landmark predictions using the software Slicer and the SlicerMorph pluggin manually if you want, 
otherwise, we developped a script for running ALPACA in batch, with varying alpha and beta parameters (to have several ALPACA predictions replicates efficiently)
Please see the code and procedure at https://github.com/charleneguillaumot/ALPACA_from_terminal

### Step 1. Rework initial ply files 

### Step 2. 
