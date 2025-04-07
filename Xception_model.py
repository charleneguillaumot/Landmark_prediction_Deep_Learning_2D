# Usage:
# python Xception_model.py --job F5_rep1 --Landmark F_5 --rep rep1
 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, Conv2DTranspose, Flatten, Activation
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import random 
import re 
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import shutil
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.regularizers import l1_l2
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Xception model ! Landmark specific models")
    parser.add_argument("--job", type=str, default="F_5_rep1", help="Job part name")
    parser.add_argument("--Landmark", type=str, default="F_48", help="Landmark name (eg 'F_18')")
    #parser.add_argument("--rep", type=str, default="rep_1", help="Model replicate")

    args = parser.parse_args()
    
    ### PARAMETERS AND PATHS 
    LDS_SELECT = args.Landmark  
    job_part = args.job
    #rep = args.rep
    JOB =f"Xception_model_{job_part}"
    
    chemin= os.getcwd()
    working_dir = os.path.join(chemin, f"DATA/cranes_souris/AREAS_AROUND_LANDMARKS_cranes_souris/")
    chemins_rasters= os.path.join(working_dir, f"LDS_{LDS_SELECT}/RASTERS_tif/")
    chemin_LDS_uv_coords= os.path.join(working_dir, f"LDS_{LDS_SELECT}/LDS_uv_coords/")
    
    # Directories to training data 
    repertoire_images_AO = f"{chemins_rasters}/AO/"
    repertoire_images_VO = f"{chemins_rasters}/VO/"
    repertoire_images_CUR = f"{chemins_rasters}/CUR1/"
    # Associated list of files 
    fichiers_AO = set(os.listdir(repertoire_images_AO))
    fichiers_VO = set(os.listdir(repertoire_images_VO))
    fichiers_CUR = set(os.listdir(repertoire_images_CUR))
    
    # TEST DATASET CREATION
    #-------------------------------------------------------------------------------------------------
    # The test dataset is created in the Xception code, after what there is no need to duplicate this in the other algorithms codes 
    
    # # Similar patterns between the folders :
    # pattern = re.compile(r'indi_(\d+-\d+)')
    
    # # Dictionary to store one filename per unique individual
    # selected_files = defaultdict(list)
    
    # # Analyse the files and store all possible individuals 
    # for filename in fichiers_VO:
    #     match = pattern.search(filename)
    #     if match:
    #         individual_id = match.group(1)
    #         selected_files[individual_id].append(filename)
    
    # selected_filenames = []
    
    # # Takes one individual out of the 3 possible 
    # for individual_id, files in selected_files.items():
    #     if len(files) >= 3:  # ????
    #         selected_files_sample = random.sample(files, 3)
    #         selected_filenames.extend(selected_files_sample)
    #     else:
    #         selected_filenames.extend(files)  # Si moins de 2 fichiers, les ajouter tous
    
    # fichiers_VO = selected_filenames
    
    # # Remove extensions _AO.tif, _CUR.tif...
    # fichiers_AO_sans_ext = [fichier.replace("_AO.tif", "") for fichier in fichiers_AO]
    # fichiers_VO_sans_ext  = [fichier.replace("_VO.tif", "") for fichier in fichiers_VO]
    # fichiers_CUR_sans_ext  = [fichier.replace("_CUR1.tif", "") for fichier in fichiers_CUR]
    # fichiers_AO_sans_ext_set = set(fichiers_AO_sans_ext)
    # fichiers_VO_sans_ext_set  = set(fichiers_VO_sans_ext)
    # fichiers_CUR_sans_ext_set  = set(fichiers_CUR_sans_ext)
    
    # chemin_dossier = f"{chemins_rasters}/TEST_DATASET/"
    # if not os.path.exists(chemin_dossier):
    #     os.makedirs(chemin_dossier)
    #     print("NO existing TEST folder, let's create one")
    #     chemin_dossier_AO = f"{chemins_rasters}/TEST_DATASET/AO/"
    #     chemin_dossier_VO = f"{chemins_rasters}/TEST_DATASET/VO/"
    #     chemin_dossier_CUR = f"{chemins_rasters}/TEST_DATASET/CUR1/"
    #     os.makedirs(chemin_dossier_AO)
    #     os.makedirs(chemin_dossier_VO)
    #     os.makedirs(chemin_dossier_CUR)
        
    #     # Extract AO, VO and CUR images from the previous selection 
    #     images_a_copier = fichiers_AO_sans_ext_set.intersection(fichiers_VO_sans_ext_set, fichiers_CUR_sans_ext_set)
        
    #     # Copy the files in the TEST folders, classified by color names 
    #     for image in images_a_copier:
    #         source_AO = os.path.join(repertoire_images_AO, image +"_AO.tif")
    #         destination = os.path.join(chemin_dossier_AO, image +"_AO.tif")
    #         shutil.copyfile(source_AO, destination)
        
    #         source_VO = os.path.join(repertoire_images_VO, image+"_VO.tif")
    #         destination = os.path.join(chemin_dossier_VO, image+"_VO.tif")
    #         shutil.copyfile(source_VO, destination)
        
    #         source_CUR = os.path.join(repertoire_images_CUR, image +"_CUR1.tif")
    #         destination = os.path.join(chemin_dossier_CUR, image +"_CUR1.tif")
    #         shutil.copyfile(source_CUR, destination)
            
    #         # Remove duplicated and transfered images 
    #         os.remove(source_AO)
    #         os.remove(source_VO)
    #         os.remove(source_CUR)
    #         print("il faudra aussi penser à commenter cette partie pour le ResNet et VIT?")
    # else:
    #     print("TEST folder already compiled")
        
    
    #-------------------------------------------------------------------------------------------------
    # The 3 types of images (AO, VO, CUR) where compiled separately, so it is possible that some are missing
    # Here, we make sure that we only keep the images for which we do have the RGB images for the 3 colors (AO, VO et CUR)
    #-----------------------------------------------------------------------------------------------------------
    # Refresh training images loading
    repertoire_images_AO = f"{chemins_rasters}/AO/"
    repertoire_images_VO = f"{chemins_rasters}/VO/"
    repertoire_images_CUR = f"{chemins_rasters}/CUR1/"
    
    fichiers_AO = set(os.listdir(repertoire_images_AO))
    fichiers_VO = set(os.listdir(repertoire_images_VO))
    fichiers_CUR = set(os.listdir(repertoire_images_CUR))
    
    # Remove extensions _AO.tif, _CUR.tif...
    fichiers_AO_sans_ext = [fichier.replace("_AO.tif", "") for fichier in fichiers_AO]
    fichiers_VO_sans_ext  = [fichier.replace("_VO.tif", "") for fichier in fichiers_VO]
    fichiers_CUR_sans_ext  = [fichier.replace("_CUR1.tif", "") for fichier in fichiers_CUR]
    
    ### SORTING images, keep only the ones with the 3 colors homologies 
    en_trop_AO_VO = set(fichiers_AO_sans_ext) - set(fichiers_VO_sans_ext)
    en_trop_AO_VO
    en_trop_VO_AO = set(fichiers_VO_sans_ext) - set(fichiers_AO_sans_ext)
    en_trop_VO_AO
    
    print ("INITIALLY")
    print (f'Folder AO size: {len(fichiers_AO)}')
    print (f'Folder VO size: {len(fichiers_VO)}')
    print (f'Folder CUR size: {len(fichiers_CUR)}')
    
    extension = "_AO.tif"
    if en_trop_AO_VO:
        for i in range(len(en_trop_AO_VO)):
            fichiers_AO.remove(list(en_trop_AO_VO)[i] + extension)
            fichiers_AO_sans_ext.remove(list(en_trop_AO_VO)[i])
       
    extension = "_VO.tif"
    if en_trop_VO_AO:
        for i in range(len(en_trop_VO_AO)):
            fichiers_VO.remove(list(en_trop_VO_AO)[i] + extension)
            fichiers_VO_sans_ext.remove(list(en_trop_VO_AO)[i])
        
    print ("1. STEP 1 ")
    print (f'Folder AO size: {len(fichiers_AO)}')
    print (f'Folder VO size: {len(fichiers_VO)}')
    print (f'Folder CUR size: {len(fichiers_CUR)}')
    
    en_trop_AO_CUR = set(fichiers_AO_sans_ext) - set(fichiers_CUR_sans_ext)
    en_trop_AO_CUR
    en_trop_CUR_AO = set(fichiers_CUR_sans_ext) - set(fichiers_AO_sans_ext)
    en_trop_CUR_AO
    
    extension = "_AO.tif"
    if en_trop_AO_CUR:
        for i in range(len(en_trop_AO_CUR)):
            fichiers_AO.remove(list(en_trop_AO_CUR)[i] + extension)
            fichiers_AO_sans_ext.remove(list(en_trop_AO_CUR)[i])
        
    extension = "_CUR1.tif"
    if en_trop_CUR_AO:
        for i in range(len(en_trop_CUR_AO)):
            fichiers_CUR.remove(list(en_trop_CUR_AO)[i] + extension)
            fichiers_CUR_sans_ext.remove(list(en_trop_CUR_AO)[i])
        
    print ("2. STEP 2")
    print (f'Folder AO size: {len(fichiers_AO)}')
    print (f'Folder VO size: {len(fichiers_VO)}')
    print (f'Folder CUR size: {len(fichiers_CUR)}')
    
    en_trop_VO_CUR = set(fichiers_VO_sans_ext) - set(fichiers_CUR_sans_ext)
    en_trop_VO_CUR
    en_trop_CUR_VO = set(fichiers_CUR_sans_ext) - set(fichiers_VO_sans_ext)
    en_trop_CUR_VO
    
    extension = "_VO.tif"
    if en_trop_VO_CUR:
        for i in range(len(en_trop_VO_CUR)):
            fichiers_VO.remove(list(en_trop_VO_CUR)[i] + extension)
            fichiers_VO_sans_ext.remove(list(en_trop_VO_CUR)[i])
        
    extension = "_CUR1.tif"
    if en_trop_CUR_VO:
        for i in range(len(en_trop_CUR_VO)):
            fichiers_CUR.remove(list(en_trop_CUR_VO)[i] + extension)
            fichiers_CUR_sans_ext.remove(list(en_trop_CUR_VO)[i])
    
    print ("3. STEP 3")
    print (f'Folder AO size: {len(fichiers_AO)}')
    print (f'Folder VO size: {len(fichiers_VO)}')
    print (f'Folder CUR size: {len(fichiers_CUR)}')
    # At the end of this part, variables fichiers_AO, fichiers_VO, fichiers_CUR contain all cleaned and equilibrated training data 
    
    ##--------------------------------------------------
    # LOAD TRAINING DATASET
    ##--------------------------------------------------
    images = []
    labels = []
    lds_uv_coords_associated = []
    
    for fich_AO in fichiers_AO:
        chemin_image_AO = os.path.join(repertoire_images_AO, fich_AO)
        image_AO = Image.open(chemin_image_AO)
        image_AO = np.array(image_AO)
        image_AO[(image_AO < -3.3e+38)&(image_AO > -3.5e+38)] = -9999
        
        # Associated image label 
        label_corres = re.sub(".tif", "", fich_AO)
        labels.append(label_corres) # store the label in a list
        
        # Associated manual landmark position 
        fichier_name = re.sub(".tif", "", fich_AO)
        fichier_name = re.sub("raster_lscm_", "", fichier_name)
        path_lds_uv_coords = f"{chemin_LDS_uv_coords}/{fichier_name}.txt"
        lds_uv_coords = np.loadtxt(path_lds_uv_coords, delimiter=',')
        x_coord = lds_uv_coords[0] *224
        y_coord = 224-lds_uv_coords[1] *224
        lds_uv_coords_up = [x_coord, y_coord]
        lds_uv_coords_associated.append(lds_uv_coords_up) # store the landmark coordinates in a list
        
        # Name of this image, to get the paired VO and CUR images 
        nom_fichier = fich_AO.replace("AO.tif", "") 
        
        # Get VO images 
        for fich_VO in fichiers_VO: 
            if nom_fichier in fich_VO:  # extract VO image with similar name as the previous AO one
                chemin_image_VO = os.path.join(repertoire_images_VO, fich_VO)
                image_VO = Image.open(chemin_image_VO)
                image_VO = np.array(image_VO)
                image_VO[(image_VO < -3.3e+38)&(image_VO > -3.5e+38)] = -9999
    
        for fich_CUR in fichiers_CUR: 
            if nom_fichier in fich_CUR:  # extract CUR image with similar name as the previous AO one
                chemin_image_CUR = os.path.join(repertoire_images_CUR, fich_CUR)
                image_CUR = Image.open(chemin_image_CUR)
                image_CUR = np.array(image_CUR)
                image_CUR[(image_CUR < -3.3e+38)&(image_CUR > -3.5e+38)] = -9999
    
        image_rgb = np.stack((image_AO, image_VO, image_CUR), axis=-1) # stack the 3 images (AO, VO, CUR) in a RGB stack
        images.append(image_rgb) #update the "image" list
        
    array_lds_uv_coords = np.vstack(lds_uv_coords_associated)
    
    # Convert lists to numpy tables 
    images = np.array(images)
    labels = np.array(labels)
    array_lds_uv_coords_labelled = np.column_stack((array_lds_uv_coords,labels)) # on colle les labels aux positions 
    
    print(images)
    print(labels)
    print(array_lds_uv_coords_labelled)
    
    #-----------------------------------
    ### LOAD TEST DATASET
    #-----------------------------------
    images_test = []
    labels_test = []
    lds_uv_coords_associated_test = []
    
    # Paths to test folders 
    repertoire_images_AO_test = f"{chemins_rasters}/AO/"
    repertoire_images_VO_test = f"{chemins_rasters}/VO/"
    repertoire_images_CUR_test = f"{chemins_rasters}/CUR1/"
    fichiers_AO_test = set(os.listdir(repertoire_images_AO_test))
    fichiers_VO_test = set(os.listdir(repertoire_images_VO_test))
    fichiers_CUR_test = set(os.listdir(repertoire_images_CUR_test))
    
    for fich_AO_test in fichiers_AO_test:
        chemin_image_AO_test = os.path.join(repertoire_images_AO_test, fich_AO_test)
        image_AO_test = Image.open(chemin_image_AO_test)
        image_AO_test = np.array(image_AO_test)
        image_AO_test[(image_AO_test < -3.3e+38)&(image_AO_test > -3.5e+38)] = -9999
        
        # Associated image label 
        label_corres_test = re.sub(".tif", "", fich_AO_test)
        labels_test.append(label_corres_test)
       
        # Associated manual landmark position
        fichier_name = re.sub(".tif", "", fich_AO_test)
        fichier_name = re.sub("raster_lscm_", "", fichier_name)
        path_lds_uv_coords = f"{chemin_LDS_uv_coords}/{fichier_name}.txt"
        lds_uv_coords = np.loadtxt(path_lds_uv_coords, delimiter=',')
        x_coord = lds_uv_coords[0] *224
        y_coord = 224-lds_uv_coords[1] *224
        lds_uv_coords_up = [x_coord, y_coord]
        lds_uv_coords_associated_test.append(lds_uv_coords_up)
        
        # Name of this image, to get the paired VO and CUR images 
        nom_fichier_test = fich_AO_test.replace("AO.tif", "") 
        
        for fich_VO_test in fichiers_VO_test: # extract VO image with similar name as the previous AO one
            if nom_fichier_test in fich_VO_test:  
                chemin_image_VO_test = os.path.join(repertoire_images_VO_test, fich_VO_test)
                image_VO_test = Image.open(chemin_image_VO_test)
                image_VO_test = np.array(image_VO_test)
                image_VO_test[(image_VO_test < -3.3e+38)&(image_VO_test > -3.5e+38)] = -9999
    
        for fich_CUR_test in fichiers_CUR_test:
            if nom_fichier_test in fich_CUR_test:  
                # Load CUR images 
                chemin_image_CUR_test = os.path.join(repertoire_images_CUR_test, fich_CUR_test)
                image_CUR_test = Image.open(chemin_image_CUR_test)
                image_CUR_test = np.array(image_CUR_test)
                image_CUR_test[(image_CUR_test < -3.3e+38)&(image_CUR_test > -3.5e+38)] = -9999
    
        image_rgb_test = np.stack((image_AO_test, image_VO_test, image_CUR_test), axis=-1)
        images_test.append(image_rgb_test) 
        
    array_lds_uv_coords_test = np.vstack(lds_uv_coords_associated_test)
    
    # Convert lists into Numpy tables 
    images_test = np.array(images_test)
    labels_test = np.array(labels_test)
    array_lds_uv_coords_test_labelled = np.column_stack((array_lds_uv_coords_test,labels_test)) 
    
    print(images_test)
    print(labels_test)
    print(array_lds_uv_coords_test_labelled)
     
    # Plot some images 
    for test_nb in range(5):
        plt.imshow(images_test[test_nb],cmap='gray', interpolation='none')
        coords_y_test = array_lds_uv_coords_test[test_nb,] # vraies positions 
    
        label = labels_test[test_nb]
        print(label)
        plt.scatter(coords_y_test[0], coords_y_test[1], color="red", label="CONNUES")
      
        plt.legend()
        plt.title(f"Label: {label}")
        plt.savefig(f'{chemin}/RESULTATS/plot_{JOB}_{label}.png')  # ou plt.savefig(f'RESULTATS/plot_{test_nb}.pdf')
        
        plt.close()
        
    
    ###--------------------------------------------------------------------------------------------------------------------
    #                                                       MODEL
    ###--------------------------------------------------------------------------------------------------------------------
    
    #---------------------------------
    # TRAINING AND VALIDATION SETS
    #---------------------------------
    X_train, X_val, y_train, y_val = train_test_split(images, array_lds_uv_coords_labelled, test_size=0.2)
    
    # Convertir les listes en tableaux NumPy
    X_train_images = np.array(X_train)
    X_val_images = np.array(X_val)
    X_train_lds_uv_coords = np.array(y_train)
    X_train_lds_uv_coords = np.array(pd.DataFrame(y_train[:,:2])).astype(float)
    
    X_train_labels = y_train[:,2].tolist() # Labels for training set
    X_val_labels = y_val[:,2].tolist()  # Labels for validation set
    
    y_val_coords = y_val[:,:2].astype(float)  # Labels for validation set
    
    # Normalize
    replaceby0 = (X_train_images == -9999)
    X_train_images[replaceby0] = 0
    replaceby0 = (X_val == -9999)
    X_val[replaceby0] = 0
    replaceby0 = (images_test == -9999)
    images_test[replaceby0] = 0
        
    
    #------------------------
    # MODEL ARCHITECTURE
    #------------------------
    from tensorflow.keras.applications import Xception
    base_model = Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    
    #model.summary()
     
    # Open to training the loaded base model architecture
    base_model.trainable = True
    
    # Extra layers added for landmark predictions 
    model = keras.Sequential([
        base_model,
        layers.MaxPooling2D((4,4)),
        Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.2, l2=0.2)),
        layers.BatchNormalization(),
        Dropout(0.5),
        layers.Dense(2, kernel_regularizer=l1_l2(l1=0.1, l2=0.1))
    ])
    
    # Compile the model 
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    #------------------------
    # CALLBACKS
    #------------------------
    early_stopping = EarlyStopping(monitor='loss', patience=70, restore_best_weights=True) # permet de rajouter un callback quand la performance du modèle en cours d'apprentissage s'arrête de grimper
    save_csv_log = f'{chemin}/run/logs/training_log_{JOB}'
    
    csv_logger = CSVLogger(save_csv_log, separator=',', append=False)
    
    save_dir = f"{chemin}/run/models/best_model_{JOB}.h5"
    checkpoint = ModelCheckpoint(f"{chemin}/run/models/best_model_{JOB}.h5",
                                  monitor='val_loss',  # Monitor loss values 
                                  save_best_only=True,  # Save only if best values are reached 
                                  mode='min',  # best -> if minimum values are observed 
                                  verbose=1)
    #------------------------
    # DATA AUGMENTATION
    #------------------------
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        #width_shift_range=[-20, 20],
        #height_shift_range=[-20, 20],
        #zoom_range=[0.5, 1.5],
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='constant')
    
    #------------------------
    # TRAIN THE MODEL
    #------------------------
    model.fit(train_datagen.flow(X_train_images, X_train_lds_uv_coords, batch_size=128), 
              epochs=1000,
              callbacks=[early_stopping, csv_logger, checkpoint],
              validation_data=(X_val, y_val_coords))
    
    #------------------------
    # MODEL PREDICTIONS
    #------------------------
    predictions = model.predict(images_test)
    x_coord = predictions[:, 0]/224
    y_coord = (224-predictions[:,1]) /224
    pred = np.column_stack((x_coord, y_coord))
       
    results_pred = pd.DataFrame(np.column_stack((predictions,labels_test)))
    
    pred_filename = f"{JOB}_predictions"
    chemin_pred= f"{chemin}/RESULTATS/{pred_filename}.csv"
   
    results_pred.to_csv(chemin_pred, index=False)
    
    # Plot some images 
    for pred_nb in range(5):
        plt.imshow(images_test[pred_nb],cmap='gray', interpolation='none')
        coords_y_test = array_lds_uv_coords_test[pred_nb,] # vraies positions 
    
        label = labels_test[pred_nb]
        print(label)
        plt.scatter(coords_y_test[0], coords_y_test[1], color="red", label="CONNUES")
        plt.scatter(pred[0], pred[1], color="blue", label="PREDICTIONS")

        plt.legend()
        plt.title(f"Label: {label}")
        plt.savefig(f'{chemin}/RESULTATS/predictions_{JOB}_{label}.png')  
        
        plt.close()
