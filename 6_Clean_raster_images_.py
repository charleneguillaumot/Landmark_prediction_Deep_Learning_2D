# Usage:
# python3 5_Clean_raster_images.py --mode parallel --cores 4 --Landmark F-18

import os
import argparse
import numpy as np
from PIL import Image
from shutil import move
from multiprocessing import Pool, cpu_count

def is_mostly_black(image_path, threshold=99):
    # Check if an image is mostly black based on a pixel intensity threshold
    try:
        img = Image.open(image_path)
        # open and convert to gray scale 
        # img = Image.open(image_path).convert('L') -> detect 0 ?
        img_array = np.array(img)
        # Count pixels with the specific black value
        count_black = np.sum(img_array == -3.3999999521443642e+38) 
        prop_black = 100 * count_black / img_array.size
        return prop_black > threshold
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def delete_black_image(image_path):
    # Deletes a black image if it meets the threshold
    if is_mostly_black(image_path):
        os.remove(image_path)
        #print(f"Deleted: {os.path.basename(image_path)}")

def delete_black_images_parallel(directory, num_cores):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tif")]
    with Pool(processes=num_cores) as pool:
        pool.map(delete_black_image, image_files)
        #print(f"Deleted: {os.path.basename(image_files)}")



def delete_black_images_sequential(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            delete_black_image(os.path.join(directory, filename))
            #print(f"Deleted: {os.path.basename(filename)}")

def organize_files(directory):
    # Move images into corresponding folders based on extension
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            for extension, folder in extensions_and_folders.items():
                if extension in file:
                    output_dir = os.path.join(directory, folder)
                    os.makedirs(output_dir, exist_ok=True)
                    move(file_path, os.path.join(output_dir, file))
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TIFF images: delete black ones and organize the rest.")
    parser.add_argument("--mode", choices=["sequential", "parallel"], default="sequential",
                        help="Choose execution mode: 'sequential' or 'parallel' (default: parallel)")
    parser.add_argument("--cores", type=int, default=cpu_count()-1,
                        help="Number of CPU cores to use (default: max available-1)")
    parser.add_argument("--Landmark", type=str, default="F_34", help="Landmark name (eg 'F_34')")

    args = parser.parse_args()

    print(f"Running in {args.mode} mode with {args.cores} cores...")

    # Working directories and files
    #"/work/crct/cgq19064/Ecolplus"
    LDS = args.Landmark # "F_34" 
    path = os.getcwd()
    working_dir = os.path.join(path, f"data/cranes_souris/AREAS_AROUND_LANDMARKS_cranes_souris/")
    raster_dir = os.path.join(working_dir, f"LDS_{LDS}/RASTERS_tif/")


     # Extensions and corresponding folders
    extensions_and_folders = {
      "_AO.tif": "AO",
      "_VO.tif": "VO",
      "_CUR1.tif": "CUR1"
      }

    # Step 1: Delete black images
    if args.mode == "parallel":
        args.cores = max(1, min(cpu_count()-1, args.cores))
        delete_black_images_parallel(raster_dir, args.cores)
    else:
        delete_black_images_sequential(raster_dir)

    # Step 2: Organize remaining images
    organize_files(raster_dir)

    print("Processing complete.")
