# Usage:
# python 6_Landmark_uv_coords.py --mode parallel --cores 4 --species hamsters --landmark F_5

import os
import re
import multiprocessing
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Function to process a single raster file
def process_raster(rasters_file):
    print(rasters_file)

    # Ouverture du subset de mesh coloré d'intérêt
    mesh_colore_corres = re.sub(r"raster_lscm_", "", rasters_file)
    mesh_colore_corres = os.path.splitext(mesh_colore_corres)[0]
 
    with open(f"{working_dir}/{mesh_colore_corres}.obj", "r") as file:
        lines = file.readlines()
    v_lines = [line.replace("v", "").strip() for line in lines if line.startswith("v ")]
    v_3d = np.array([list(map(float, line.split())) for line in v_lines])
    v_3d = v_3d[:,0:3]
    
    # open lds files
    motif = r"\d{3}-\d{2}|\d{3}-\d{1}"
    match = re.search(motif, mesh_colore_corres)
    if match:
        lds_initiaux = pd.read_csv(
                os.path.join(folder_lds, f"{match.group()}.fcsv"),
                header=None,
                comment='#',
                names=["id", "x", "y", "z", "ow", "ox", "oy", "oz", "vis", "sel", "lock", "label", "desc", "associatedNodeID", "NA1", "NA12"],
                sep=","
            )
        coords_lds_3d = lds_initiaux[lds_initiaux['label']==LDS_SELECT][["x","y","z"]]
        print(coords_lds_3d)
        # Search of the closest mesh vertex to the Landmark 
        tree = cKDTree(v_3d)
        _, nearest_neighbor_index = tree.query(coords_lds_3d)
        print(nearest_neighbor_index)

        # Search of the UV coordinates of the closest vertex
        # --------
        with open(f"{uv_dir}{mesh_colore_corres}_lscm.obj", "r") as file:
            OBJ_param_LSCM = file.readlines()

        vt_lines = [line for line in OBJ_param_LSCM if line.startswith("vt ")]
        vt = [list(map(float, line.split()[1:3])) for line in vt_lines]

        # Coordonnées UV LDS
        coords_uv_lds = vt[nearest_neighbor_index[0]]
        with open(f"{folder_uvcoords}{mesh_colore_corres}.txt", "w") as fichier:
            fichier.write(", ".join(map(str, coords_uv_lds)))

# Function to run sequentially
def run_sequential():
    for rasters_file in rasters_list_files:
        process_raster(rasters_file)

# Function to run in parallel
def run_parallel(num_cores):
    num_workers = min(num_cores, len(rasters_list_files))  
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_raster, rasters_list_files)

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raster files in parallel or sequentially.")
    parser.add_argument("--mode", choices=["sequential", "parallel"], default="parallel",
                        help="Choose execution mode: 'sequential' or 'parallel' (default: parallel)")
    parser.add_argument("--cores", type=int, default=max(1, multiprocessing.cpu_count() - 1),
                        help="Number of CPU cores to use (default: max available-1)")

    parser.add_argument("--species", type=str, default="cranes_souris", help="Species name (e.g., 'hamsters')")
    parser.add_argument("--Landmark", type=str, default="F_34", help="Landmark name (eg 'F-18')")

    args = parser.parse_args()
    
    # Analysis and landmark of interest, defines which folder to process
    LDS_SELECT = args.Landmark # "F_5" 
    species = args.species # species = "hamster"

    # working directories and listing files to process
    # path = "/work/crct/cgq19064/Ecolplus/"
    path = os.getcwd()
    working_dir = os.path.join(path, f"data/cranes_souris/AREAS_AROUND_LANDMARKS_cranes_souris/LDS_{LDS_SELECT}/")
    folder_uvcoords = f"{working_dir}/LDS_uv_coords/"
    folder_lds = os.path.join(path, f"data/cranes_souris/Skull_Landmarks/")
    uv_dir = f"{working_dir}/param/"
    
    # Create folder for UV coordinates if it doesn't exist
    os.makedirs(folder_uvcoords, exist_ok=True)
    print("uv_coords folder exists or has been created")
    
    # Reading rasters
    pattern_col = "_AO"  # use only AO here because ply files were generated only for AO 
    rasters_list_files = [f for f in os.listdir(f"{working_dir}/RASTERS_tif/AO/") if pattern_col in f]

    if args.mode == "parallel":
        args.cores = max(1, min(multiprocessing.cpu_count()-1, args.cores))
        print(f"Running in parallel mode with {args.cores} cores...")
        run_parallel(args.cores)
    else:
        print("Running in sequential mode...")
        run_sequential()
