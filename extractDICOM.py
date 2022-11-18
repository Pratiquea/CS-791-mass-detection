#!/usr/bin/env python
# coding: utf-8

# # Figure out how to extract and rename .dcm files
# 
# **Problem:**
# 
# The .dcm files download from CBIS-DDSM are stored in a very nested folder structure that is difficult for a script to access. We need to move these files into a single folder and rename the files according to what images they are (either full mammogram scan, cropped image or mask image).

# In[1]:


import os
import shutil
import pydicom
import pandas as pd
from pathlib import Path

import numpy as np
import random


# In[2]:


top = Path("/home/cs457/Desktop/project/dataa")


# ## Helper functions

# In[3]:


def new_name_dcm(dcm_path):
    
    """
    This function takes the absolute path of a .dcm file
    and renames it according to the convention below:
    
    1. Full mammograms:
        - Mass-Training_P_00001_LEFT_CC_FULL.dcm
    2. Cropped image:
        - Mass-Training_P_00001_LEFT_CC_CROP_1.dcm
        - Mass-Training_P_00001_LEFT_CC_CROP_2.dcm
        - ...
    3. Mask image:
        - Mass-Training_P_00001_LEFT_CC_MASK_1.dcm
        - Mass-Training_P_00001_LEFT_CC_MASK_2.dcm
        - ...
    
    
    Parameters
    ----------
    dcm_path : {str}
        The relative (or absolute) path of the .dcm file
        to rename, including the .dcm filename.
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC/1-1.dcm"

    Returns
    -------
    new_name : {str}
        The new name that the .dcm file should have
        WITH the ".dcm" extention WITHOUT its relative
        (or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.dcm"
    False : {boolean}
        False is returned if the new name of the .dcm
        file cannot be determined.
    """
    
    try:
        # Read dicom.
        ds = pydicom.dcmread(dcm_path)
    
    except Exception as ex:
        print(ex)
        return None
    
    else:
        # Get information.
        patient_id = ds.PatientID
        img_type = ds.SeriesDescription

        # === FULL ===
        if "full" in img_type:
            new_name = patient_id + "_FULL" + ".dcm"
            print(f"FULL --- {new_name}")
            return new_name

        # === CROP ===
        elif "crop" in img_type:

            # Double check if suffix is integer. 
            suffix = patient_id.split("_")[-1]

            if suffix.isdigit():
                new_patient_id = patient_id.split("_" + suffix)[0]
                new_name = new_patient_id + "_CROP" + "_" + suffix + ".dcm"
                print(f"CROP --- {new_name}")
                return new_name

            elif not suffix.isdigit():
                print(f"CROP ERROR, {patient_id}")
                return False

        # === MASK ===
        elif "mask" in img_type:

            # Double check if suffix is integer. 
            suffix = patient_id.split("_")[-1]

            if suffix.isdigit():
                new_patient_id = patient_id.split("_" + suffix)[0]
                new_name = new_patient_id + "_MASK" + "_" + suffix + ".dcm"
                print(f"MASK --- {new_name}")
                return new_name


            elif not suffix.isdigit():
                print(f"MASK ERROR, {patient_id}")
                return False

        # === img_type NOT RECOGNISED ===
        else:
            print(f"img_type CANNOT BE IDENTIFIED, {img_type}")
            return False


# In[4]:


def move_dcm_up(dest_dir, source_dir, dcm_filename):

    """
    This function move a .dcm file from its given source
    directory into the given destination directory. It also
    handles conflicting filenames by adding "___a" to the
    end of a filename if the filename already exists in the
    destination directory.
    
    Parameters
    ----------
    dest_dir : {str}
        The relative (or absolute) path of the folder that
        the .dcm file needs to be moved to.
    source_dir : {str}
        The relative (or absolute) path where the .dcm file
        needs to be moved from, including the filename.
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC_FULL.dcm"
    dcm_filename : {str}
        The name of the .dcm file WITH the ".dcm" extension
        but WITHOUT its (relative or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.dcm".
        
    Returns
    -------
    None
    """
    
    dest_dir_with_new_name = os.path.join(dest_dir, dcm_filename)

    # If the destination path does not exist yet...
    if not os.path.exists(dest_dir_with_new_name):
        shutil.move(source_dir, dest_dir)

    # If the destination path already exists...
    elif os.path.exists(dest_dir_with_new_name):
        # Add "_a" to the end of `new_name` generated above.
        new_name_2 = dcm_filename.strip(".dcm") + "___a.dcm"
        # This moves the file into the destination while giving the file its new name.
        shutil.move(source_dir, os.path.join(dest_dir, new_name_2))


# In[5]:


def delete_empty_folders(top, error_dir):
    
    """
    This function recursively walks through a given directory
    (`top`) using depth-first search (bottom up) and deletes
    any directory that is empty (ignoring hidden files).
    If there are directories that are not empty (except hidden
    files), it will save the absolute directory in a Pandas
    dataframe and export it as a `not-empty-folders.csv` to
    `error_dir`.

    Parameters
    ----------
    top : {str}
        The directory to iterate through.
    error_dir : {str}
        The directory to save the `not-empty-folders.csv` to.

    Returns
    -------
    None
    """
    
    curdir_list = []
    files_list = []
    
    for (curdir, dirs, files) in os.walk(top=top, topdown=False):
    
        if curdir != str(top):

            dirs.sort()
            files.sort()

            print(f"WE ARE AT: {curdir}")
            print("=" * 10)

            print("List dir:")

            directories_list = [f for f in os.listdir(curdir) if not f.startswith('.')]
            print(directories_list)

            if len(directories_list) == 0:
                print("DELETE")
                shutil.rmtree(curdir, ignore_errors=True)

            elif len(directories_list) > 0:
                print("DON'T DELETE")
                curdir_list.append(curdir)
                files_list.append(directories_list)

            print()
            print("Moving one folder up...")
            print("-" * 40)
            print()

    if len(curdir_list) > 0:
        not_empty_df = pd.DataFrame(list(zip(curdir_list, files_list)),
                                    columns =["curdir", "files"])
        to_save_path = os.path.join(error_dir, "not-empty-folders.csv")
        not_empty_df.to_csv(to_save_path, index=False)
        


# ## 1. Count how many .dcm files before executing

# In[6]:


before = 0

# Count number of .dcm files in ../data/Mass/Test.
for rootdir, dirs, files in os.walk(top):
    for f in files:
        if f.endswith(".dcm"):
            before += 1

print(f"BEFORE --> Number of .dcm files: {before}")


# ## 2. Execute

# In[10]:


# Rename and move .dcm files.
for (curdir, dirs, files) in os.walk(top=top, topdown=False):

    dirs.sort()
    files.sort()

    print(f"WE ARE AT: {curdir}")
    print("=" * 10)
    
    for f in files:
        
        # === Step 1: Rename .dcm file ===
        if f.endswith(".dcm"):
            
            old_name_path = os.path.join(curdir, f)
            new_name = new_name_dcm(dcm_path=old_name_path)
            
            if new_name:
                new_name_path = os.path.join(curdir, new_name)
                os.rename(old_name_path, new_name_path)
        
                # === Step 2: Move RENAMED .dcm file ===
                move_dcm_up(dest_dir="/home/cs457/Desktop/project/dataaa", source_dir=new_name_path, dcm_filename=new_name)
    
    print()
    print("Moving one folder up...")
    print("-" * 40)
    print()


# In[11]:


# Delete empty folders.
delete_empty_folders(top=top, error_dir=top)


# ## 3. Count how many .dcm files after executing

# In[12]:


after = 0

# Count number of .dcm files in ../data/Mass/Test.
for rootdir, dirs, files in os.walk(top):
    for f in files:
        if f.endswith(".dcm"):
            after += 1

print(f"AFTER --> Number of .dcm files: {after}")


# ## Settle extracting of calcification test .dcm

# In[13]:


#top = "../data/raw_data/Calc/Test"


# In[30]:


#before = 0
#for rootdir, dirs, files in os.walk(top):
#    for f in files:
#        if f.endswith(".dcm"):
#            before += 1

#print(before)


# In[15]:


#top = "../dataaa/"
#extension = ".png"

#mass = []
#calc = []

#for (curdir, dirs, files) in os.walk(top=top, topdown=False):

#    dirs.sort()
#    files.sort()

#    for f in files:

 #       if f.endswith(extension):
            
#            if "mass" in f.lower():
#                mass.append(f)
  #          elif "calc" in f.lower():
 #               calc.append(f)


# In[32]:


#split = 0.2
#mass_test_count = round(split * len(mass))
#calc_test_count = round(split * len(calc))

#mass_test = random.sample(mass, mass_test_count)
#mass_train = [m for m in mass if m not in mass_test]

#calc_test = random.sample(calc, calc_test_count)
#calc_train = [c for c in calc if c not in calc_test]

#test = mass_test + calc_test
#train = mass_train + calc_train


# In[29]:


#test_df = pd.


# In[30]:


# mass2 = [m.replace("Mass_", "") for m in mass]
# calc2 = [c.replace("Calc_", "") for c in calc]

# mass3 = [m.replace("_FULL___PRE.png", "") for m in mass2]
# calc3 = [c.replace("_FULL___PRE.png", "") for c in calc2]

# mass_set = set(mass3)
# calc_set = set(calc3)
# intersection = mass_set & calc_set

# to_remove_mass = [m for m in mass if any(inter in m for inter in intersection)]
# to_remove_calc = [c for c in calc if any(inter in c for inter in intersection)]

# to_remove = to_remove_mass + to_remove_calc

# for (curdir, dirs, files) in os.walk(top=top, topdown=False):

#     dirs.sort()
#     files.sort()
    
#     for f in files:
        
#         if f.endswith(".png") and f in to_remove:
#             os.remove(os.path.join(curdir, f))


# In[ ]:





# In[ ]:





# In[ ]:




