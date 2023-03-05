import numpy as np
import os
# %%
"""
# Import
"""

# %%
import numpy as np

import matplotlib.pyplot as plt









import io

import cv2

import imageio

#from ipywidgets import widgets, Layout, HBox

import matplotlib.colors as colors

from skimage.transform import resize







import os



# %%
"""
# Functions
"""

# %%
"""

"""

# %%
import re

def sorted_alphanumeric(data):

    convert = lambda text: int(text) if text.isdigit() else text.lower()

    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 

    return sorted(data, key=alphanum_key)

# %%
# load all images from a givel folder

# load all images from a givel folder
def load_images_from_folder(folder):
    images = []
    list=sorted_alphanumeric(os.listdir(folder))

    for filename in list:

        img = cv2.imread(os.path.join(folder,filename))  # if binary, cv2.imread(os.path.join(folder,filename),2)

        img=cv2.resize(img, (128,128))
        # img=img[:,:,0]                #  uncomment if the desired output is  binary images
        # img=img.reshape(128,128,1)    #  idm

        #ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)  # uncomment if binary images

        if img is not None:

            images.append(img)

    return images

# %%
# check if all elements of a list are equals or not

def are_equal(list):

    nTemp = list[0]

    bEqual = True

    

    for item in list:

        if nTemp != item:

            bEqual = False

            break;      

    return bEqual

# %%
# get dimensions of the data

def get_data_shape(input_dir) :

    data=[]

    # number of folders in current directory

    folders_number=len([name for name in os.listdir(input_dir) ])

    # get length of each sequence

    sub_directories = os.listdir(input_dir) 

    array_seq_len=[]

    for f in sub_directories:

        folder_list = sorted_alphanumeric(os.listdir(os.path.join(input_dir, f)))

        f_path=os.path.join(input_dir,f)

        seq_len=len([filename for filename in sorted_alphanumeric(os.listdir(f_path)) if os.path.isfile(os.path.join(f_path, filename))])

        array_seq_len.append(seq_len)

    # get image_size

    im = cv2.imread(os.path.join(f_path, os.listdir(f_path)[0]))

    #  check if all sequences have the same length

    if are_equal(array_seq_len):

        data_shape=np.array([folders_number, array_seq_len[0],im.shape[0],im.shape[1],im.shape[2]])

        return data_shape

    else:

        print('error, check your folders, at least one of them contains a different number of images') 

# %%
#@jit(nopython=True)

def load_all_data(input_dir):

    #data_shape = get_data_shape(input_dir) 

    #np.empty(shape=(data_shape[0],data_shape[1],data_shape[2],data_shape[3],data_shape[4] ))  

    #data=np.empty([data_shape[1],data_shape[2],data_shape[3],data_shape[4] ])  

    data=[]

    folders = sorted_alphanumeric(os.listdir(input_dir))

    for f in folders:

        f_path=os.path.join(input_dir,f)

        folder_list = [filename for filename in sorted_alphanumeric(os.listdir(f_path)) if os.path.isfile(os.path.join(f_path, filename))]

        print(f_path)

        sequence= load_images_from_folder(os.path.join(input_dir, f))

        data.append(sequence)      

    data=np.asarray(data)    

    #data = data.reshape(data_shape[0],data_shape[1],...)

    #print('data imported, seq_number:= ',data.shape[0])

    #data = np.expand_dims(data, axis=-1)   #uncomment if binary

    return data


#data_dir='/scratch/ulg/msm/sfetni/LSTM/sim_data_1000' 
data_dir='/CECI/trsf/sim_data_1000'


import time
start_time = time.time()

print('loading data')
dataset=load_all_data(data_dir)
print("data imported in :--- %s hours ---" % ((time.time() - start_time)/3600))


np.save('dataset_10000.npy', dataset)
