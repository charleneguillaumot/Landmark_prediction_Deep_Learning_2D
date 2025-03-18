# 1_Rework_initial_ply_files.ply using parallel processing
#
# Each mesh will be colored according to ambient occlusion 
# later AO will be used to separate inside and outside parts of meshes 
# Need to be done only once on the big 3D meshes 
# Usage:
# python 1_Rework_initial_ply_files.py --mode parallel --cores 4 --species hamsters --bone Skull

import pymeshlab
import os
import argparse
import multiprocessing

# Function to process a single mesh
def process_mesh(filename, mesh_folder):
    if "AO" in filename:
        return
    new_filename = os.path.join(mesh_folder, os.path.splitext(filename)[0] + "_AO.ply")
    if os.path.exists(new_filename):
        return

    print(f"Processing: {filename}")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(mesh_folder, filename))
           
    ms.apply_filter("compute_scalar_ambient_occlusion")
    ms.save_current_mesh(new_filename)

# Function to run sequentially
def run_sequential():
    for file in mesh_files:
        process_mesh(file, mesh_folder)

# Function to run in parallel
def run_parallel(num_workers):
    num_workers = min(multiprocessing.cpu_count()-1, num_workers) 
    print(f"Using {num_workers} workers for parallel processing...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        arguments = [(filename, mesh_folder) for filename in mesh_files]  
        pool.starmap(process_mesh, arguments) 

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 3D meshes in parallel or sequentially.")
    parser.add_argument("--mode", choices=["sequential", "parallel"], default="sequential",
                        help="Choose execution mode: 'sequential' or 'parallel' (default: parallel)")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count()-1,
                        help="Number of CPU cores to use (default: max available -1)")
    parser.add_argument("--species", type=str, default="souris", help="Species name (e.g., 'hamsters')") #required=True
    parser.add_argument("--bone", type=str, default="Skull", choices=["Mandible", "Skull"], help="Bone type (Mandible or Skull)") #required=True
    
    args = parser.parse_args()

    # path definition
    working_path = "C:/Users/ch1371gu/Desktop/POSTE BIOGEOSCIENCES/Projet Ecol+/MAPPING/PIPELINE_CODES/pipeline_codes_repris_Nico/"
    species = args.species # species = "hamsters"
    bone = args.bone       # bone = "Skull" # Mandible/Skull
    
    mesh_folder = f'{working_path}data/cranes_{species}/{bone}_Surfaces_total/'

    # Get list of mesh files
    mesh_files = [f for f in os.listdir(mesh_folder) if f.endswith(".ply") and "AO" not in f]

    if args.mode == "parallel":
        args.cores = max(1, min(multiprocessing.cpu_count()-1, args.cores))
        print(f"Running in parallel mode with {args.cores} cores...")
        run_parallel(args.cores)
    else:
        print("Running in sequential mode...")
        run_sequential()

