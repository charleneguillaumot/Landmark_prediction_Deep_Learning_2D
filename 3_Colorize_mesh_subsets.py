# Usage:
# python 3_Colorize_mesh_subsets.py --mode parallel --cores 4 --species hamsters --Landmark F-18

import pymeshlab
import os
import multiprocessing
import argparse

# Function to process a single mesh
def process_mesh(filename, working_path, lds_dir, colorless_dir):
    file_name = os.path.splitext(filename)[0]
    print(f"Processing {file_name}")
    
    #Check if the output files already exist
    ao_file = os.path.join(lds_dir, file_name + "_AO.obj")
    cur_file = os.path.join(lds_dir, file_name + "_CUR1.obj")
    vo_file = os.path.join(lds_dir, file_name + "_VO.obj")

    if os.path.exists(ao_file) and os.path.exists(cur_file) and os.path.exists(vo_file):
       print(f"Files for {file_name} already exist. Skipping.")
       return  # Skip processing if files exist
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(colorless_dir, filename))
    # CHECK if mesh manifoldness is OK ? 
    measures = ms.apply_filter("get_topological_measures")
    if measures['is_mesh_two_manifold'] and measures['number_holes'] < 10:
        ms.apply_filter('compute_texcoord_by_function_per_vertex')
        ms.apply_filter('compute_color_by_function_per_vertex')
        ms.apply_filter("compute_texcoord_transfer_vertex_to_wedge")
         
    # CLEANING   
    ms.apply_filter('meshing_remove_duplicate_vertices')
    ms.apply_filter('compute_selection_from_mesh_border')
    ms.apply_filter('meshing_close_holes', maxholesize=20)
    
    # Close holes loop (max 10 iterations)
    for _ in range(10):
        measures = ms.apply_filter("get_topological_measures")
        if measures['number_holes'] <= 1:
            break
        ms.apply_filter('compute_selection_from_mesh_border')
        ms.apply_filter('apply_coord_hc_laplacian_smoothing')
        ms.apply_filter('meshing_close_holes', maxholesize=20)
    
    # Fix non-manifold issues (max 10 iterations)
    for _ in range(10):
        measures = ms.apply_filter("get_topological_measures")
        if measures['is_mesh_two_manifold']:
            break
        ms.apply_filter('meshing_repair_non_manifold_edges', method=0)
        ms.apply_filter('meshing_repair_non_manifold_edges', method=1)
        ms.apply_filter('meshing_repair_non_manifold_vertices', vertdispratio=1)
    
    # Check if the mesh is good for texturing
    measures = ms.apply_filter("get_topological_measures")
    if measures['is_mesh_two_manifold'] and measures['number_holes'] < 10:
        ms.apply_filter('compute_texcoord_by_function_per_vertex')
        ms.apply_filter('compute_color_by_function_per_vertex')
        ms.apply_filter("compute_texcoord_transfer_vertex_to_wedge")
    
    ms.apply_filter("meshing_remove_duplicate_vertices")
    
    # Apply final manifold repair if needed
    measures = ms.apply_filter("get_topological_measures")
    if not measures['is_mesh_two_manifold']:
        ms.apply_filter('meshing_repair_non_manifold_edges', method=0)
        ms.apply_filter('meshing_repair_non_manifold_edges', method=1)
        ms.apply_filter('meshing_repair_non_manifold_vertices', vertdispratio=1)
    
    # COLORIZATION & SAVING
    
    # Ambient Occlusion (AO)
    ms.apply_filter("compute_scalar_ambient_occlusion")
    ms.save_current_mesh(os.path.join(lds_dir, file_name + "_AO.obj"))
    
    # Curvature (CUR)
    ms.apply_filter("compute_curvature_and_color_apss_per_vertex")
    ms.save_current_mesh(os.path.join(lds_dir, file_name + "_CUR.obj"))
    
    # Volumetric Obscurance (VO)
    ms.load_filter_script(os.path.join(working_path, "volumetric_obscurance.mlx"))
    ms.apply_filter_script()
    ms.save_current_mesh(os.path.join(lds_dir, file_name + "_VO.obj"))
        
    
# Function to run sequentially
def run_sequential():
    for file in files:
        process_mesh(file, working_path, lds_dir, colorless_dir)

# Function to run in parallel
def run_parallel(num_cores):
    num_workers = min(num_cores, len(files))  
    with multiprocessing.Pool(processes=num_workers) as pool:
        arguments = [(filename, working_path, lds_dir, colorless_dir) for filename in files]  
        pool.starmap(process_mesh, arguments) 

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 3D meshes in parallel or sequentially.")
    parser.add_argument("--mode", choices=["sequential", "parallel"], default="sequential",
                        help="Choose execution mode: 'sequential' or 'parallel' (default: parallel)")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count()-1,
                        help="Number of CPU cores to use (default: max available-1)")
    parser.add_argument("--species", type=str, default="cranes_souris", help="Species name (e.g., 'hamsters')")#required=True
    parser.add_argument("--Landmark", type=str, default="F_48", help="Landmark name (eg 'F-18')")#required=True

    args = parser.parse_args()
    
    # Analysis and landmark of interest, defines which folder to process
    LDS = args.Landmark # "F-18" 
    species = args.species # species = "primates"

    # Working directories and files
    #"C:/Users/ch1371gu/Desktop/POSTE BIOGEOSCIENCES/Projet Ecol+/MAPPING/"
    working_path = "C:/Users/ch1371gu/Desktop/POSTE BIOGEOSCIENCES/Projet Ecol+/MAPPING/PIPELINE_CODES/pipeline_codes_repris_Nico/"
    lds_dir = f"{working_path}data/{species}/AREAS_AROUND_LANDMARKS_{species}/LDS_{LDS}/colored/"
    colorless_dir = os.path.join(lds_dir, "colorless")

    files = [f for f in os.listdir(colorless_dir) if f.endswith('.ply')]  # Filter only mesh files
    nb_files = len(files)
    print(f"Nb of total files: {nb_files}") 

    if args.mode == "parallel":
        args.cores = max(1, min(multiprocessing.cpu_count()-1, args.cores))
        print(f"Running in parallel mode with {args.cores} cores...")
        run_parallel(args.cores)
    else:
        print("Running in sequential mode...")
        run_sequential()

    # Remove .mtl files (in both modes)
    for filename in os.listdir(lds_dir):
        if filename.endswith('.mtl'):
            os.remove(os.path.join(lds_dir, filename))
