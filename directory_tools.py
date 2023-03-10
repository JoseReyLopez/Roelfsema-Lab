import os
import random
import numpy as np



######

def create_folder(numb1, layer, extension = ''):
    folder_name = "ROI_{}_Layer_{}".format(numb1, layer)
    if not os.path.exists(folder_name):
        # If the ROI/layer folder doesn't exist yet, create it
        os.makedirs(folder_name)
    os.chdir(folder_name)

    # Create a subfolder with an index appended to the name
    if extension != '':
        subfolder_name = os.path.join('', "{}_0_{}".format(folder_name, extension))
        subfolder_index = 0
        while os.path.exists(subfolder_name):
            # If the subfolder already exists, increment the index and try again
            subfolder_index += 1
            subfolder_name = os.path.join('', "{}_{}_{}".format(folder_name, subfolder_index, extension))
    if extension == '':
        subfolder_name = os.path.join('', "{}_0".format(folder_name))
        subfolder_index = 0
        while os.path.exists(subfolder_name):
            # If the subfolder already exists, increment the index and try again
            subfolder_index += 1
            subfolder_name = os.path.join('', "{}_{}".format(folder_name, subfolder_index))

    # Create the subfolder and switch to it
    os.makedirs(subfolder_name)
    os.chdir(subfolder_name)


#########################


def clean_up_folders():
    # Get a list of all folders in the current directory
    folders = [f for f in os.listdir() if os.path.isdir(f)]

    # Iterate over the folders
    for folder in folders:
        # Change the working directory to the current folder
        os.chdir(folder)

        # Get a list of all subfolders in the current folder
        subfolders = [f for f in os.listdir() if os.path.isdir(f)]

        n_files = list(map(len, list(map(os.listdir, subfolders))))
        # Iterate over the subfolders
        for subfolder, n_file in zip(subfolders, n_files):
            # Check if the subfolder is empty
            if n_file==0:
                # If the subfolder is empty, delete it
                os.rmdir(subfolder)

        # Check if the current folder is empty
        
        os.chdir('..')
        is_empty = not len(os.listdir(folder))

        if is_empty:
            # If the current folder is empty, delete it
            os.rmdir(folder)

        # Change the working directory back to the parent directory

##########


'''
def renumber_subfolders(extension = ''):
    # Get a list of all folders in the current directory
    folders = [f for f in os.listdir() if os.path.isdir(f)]

    # Iterate over the folders
    for folder in folders:
        # Change the working directory to the current folder
        os.chdir(folder)

        # Get a list of all subfolders in the current folder
        subfolders = [f for f in os.listdir() if os.path.isdir(f)]

        # Sort the subfolders by their index values (assumes the index is the final number in the name)
        subfolders.sort(key=lambda f: int(f.split("_")[-2]))

        # Iterate over the subfolders and rename them to have consecutive index values starting from 0
        for i, subfolder in enumerate(subfolders):
            # Create a new subfolder name with the index set to the current loop iteration number
            new_name = "ROI_{}_Layer_{}_{}_{}".format(subfolder.split("_")[1], subfolder.split("_")[3], i, extension)

            # Rename the subfolder
            if not os.path.exists(new_name):
                os.rename(subfolder, new_name)

        # Change the working directory back to the parent directory
        os.chdir("..")
'''

def renumber_subfolders(extension = ''):
    # Get a list of all folders in the current directory
    folders = [f for f in os.listdir() if os.path.isdir(f)]

    # Iterate over the folders
    for folder in folders:
        # Change the working directory to the current folder
        os.chdir(folder)

        # Get a list of all subfolders in the current folder
        subfolders = [f for f in os.listdir() if os.path.isdir(f)]

        # Sort the subfolders by their index values (assumes the index is the final number in the name)        
        subfolders.sort(key=lambda f: int(f.split("_")[-2]))

        # Iterate over the subfolders and rename them to have consecutive index values starting from 0
        for i, subfolder in enumerate(subfolders):
            # Create a new subfolder name with the index set to the current loop iteration number
            new_name = "ROI_{}_Layer_{}_{}_{}".format(subfolder.split("_")[1], subfolder.split("_")[3], i, extension)

            # Rename the subfolder
            if not os.path.exists(new_name):
                os.rename(subfolder, new_name)

        # Change the working directory back to the parent directory
        os.chdir("..")


############################

def delete_all_but_lowest_validation_file():
    # Get a list of all the files in the directory that end in '.pt'
    files = [f for f in os.listdir() if f.endswith('.pt')]
    
    # Check that all the files have the same numbers in the 'model' and 'epoch' positions
    # Extract the numbers by splitting the filenames on '_' and taking the 3rd and 5th elements
    model_epoch_numbers = [tuple(f.split('_')[2:4]) for f in files]
    if not all(nums == model_epoch_numbers[0] for nums in model_epoch_numbers):
        raise ValueError("Not all files have the same model and epoch numbers")
    
    # Extract the loss values from the filenames by splitting on '_' and taking the 7th element
    loss_values = [float(f.split('_')[f.split('_').index('loss')+1]) for f in files]
    
    # Find the index of the file with the lowest loss value
    lowest_loss_index = np.argmin(loss_values)
    
    # Get the filename of the file with the lowest loss value
    lowest_validation_file = files[lowest_loss_index]
    
    # Delete all the other files
    for f in files:
        if f != lowest_validation_file:
            os.remove(f)