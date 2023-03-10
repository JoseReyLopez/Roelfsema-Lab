import os
import random


######

def create_folder(numb1, numb2):
    folder_name = "ROI_{}_Layer_{}".format(numb1, numb2)
    if not os.path.exists(folder_name):
        # If the ROI/layer folder doesn't exist yet, create it
        os.makedirs(folder_name)
    os.chdir(folder_name)

    # Create a subfolder with an index appended to the name
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



def renumber_subfolders():
    # Get a list of all folders in the current directory
    folders = [f for f in os.listdir() if os.path.isdir(f)]

    # Iterate over the folders
    for folder in folders:
        # Change the working directory to the current folder
        os.chdir(folder)

        # Get a list of all subfolders in the current folder
        subfolders = [f for f in os.listdir() if os.path.isdir(f)]

        # Sort the subfolders by their index values (assumes the index is the final number in the name)
        subfolders.sort(key=lambda f: int(f.split("_")[-1]))

        # Iterate over the subfolders and rename them to have consecutive index values starting from 0
        for i, subfolder in enumerate(subfolders):
            # Create a new subfolder name with the index set to the current loop iteration number
            new_name = "ROI_{}_Layer_{}_{}".format(subfolder.split("_")[1], subfolder.split("_")[3], i)

            # Rename the subfolder
            os.rename(subfolder, new_name)

        # Change the working directory back to the parent directory
        os.chdir("..")


############################

