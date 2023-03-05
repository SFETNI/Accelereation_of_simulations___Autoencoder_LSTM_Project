# %%
"""
# Import
"""

# %%
import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



import io

import cv2

import imageio

#from ipywidgets import widgets, Layout, HBox

import matplotlib.colors as colors

from skimage.transform import resize





from keras.callbacks import EarlyStopping

import os

from sklearn.preprocessing import MinMaxScaler

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

def load_images_from_folder(folder):

    images = []

    list=sorted_alphanumeric(os.listdir(folder))

    for filename in list:

        img = cv2.imread(os.path.join(folder,filename))  # if binary, cv2.imread(os.path.join(folder,filename),2)

        img=cv2.resize(img, (128,128))

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

        folder_list = os.listdir(os.path.join(input_dir, f))

        f_path=os.path.join(input_dir,f)

        seq_len=len([filename for filename in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, filename))])

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

    folders = os.listdir(input_dir)[:350]

    for f in folders:

        f_path=os.path.join(input_dir,f)

        folder_list = [filename for filename in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, filename))]

        #print(folder_list)

        sequence= load_images_from_folder(os.path.join(input_dir, f))

        data.append(sequence)      

    data=np.asarray(data)    

    #data = data.reshape(data_shape[0],data_shape[1],...)

    #print('data imported, seq_number:= ',data.shape[0])

    #data = np.expand_dims(data, axis=-1)   #uncomment if binary

    return data

# %%
"""
# Import data
"""

# %%
data_path ='/CECI/trsf/tmp_sf/dataset_10000.npy' 
print('start')
# %%
dataset=np.load(data_path)
print('continue')
# %%
"""
## prepare data for encoder-decoder model
"""

# %%
encod_dataset=dataset.reshape(np.prod(dataset.shape[:2]),*dataset.shape[2:])

print(dataset.shape,encod_dataset.shape)

X=encod_dataset

# %%
X = X.astype('float32') / 255.0 - 0.5

# %%
print(X.max(), X.min())

# %%
"""
## show sample(s)
"""

# %%

import matplotlib.pyplot as plt

def show_image(x):

    plt.imshow(np.clip(x + 0.5, 0, 1))

# %%
show_image(X[-1])
plt.savefig('example')
plt.close()

# %%
from sklearn.model_selection import train_test_split

# the use of random_state is to reproduce the same results at each time 

X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

# %%
print(X_train.shape)

print(X_test.shape ) 

# %%
"""
# Encoder-Decoder Model
"""

# %%
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer

from keras.models import Sequential, Model

# %%
def build_autoencoder(img_shape, code_size):

    # The encoder

    encoder = Sequential()

    encoder.add(InputLayer(img_shape))

    encoder.add(Flatten())

    encoder.add(Dense(code_size))



    # The decoder

    decoder = Sequential()

    decoder.add(InputLayer((code_size,)))

    decoder.add(Dense(np.prod(img_shape))) 

    decoder.add(Reshape(img_shape))



    return encoder, decoder

# %%
IMG_SHAPE = X.shape[1:]

encoder, decoder = build_autoencoder(IMG_SHAPE, 500)



inp = Input(IMG_SHAPE)

code = encoder(inp)

reconstruction = decoder(code)



autoencoder = Model(inp,reconstruction)

autoencoder.compile(optimizer='adamax', loss='mse', metrics=['MSE'])




print(autoencoder.summary())
callbacks = [EarlyStopping(monitor='MSE', min_delta=0, patience=10)]
# %%
"""
## train the encoder-decoder model
"""

# %%
history = autoencoder.fit(X_train, X_train,
        epochs=12,
                batch_size=264,

                shuffle=True,

                validation_data=(X_test, X_test))

# %%
save_dir=os.path.join(os.getcwd(), 'saved_computed_models')

#autoencoder.save(os.path.join(save_dir, 'encoder_decoder_big_data_1000.h5'))

encoder.save(os.path.join(save_dir, 'encoder_10000_500.h5'))

decoder.save(os.path.join(save_dir, 'decoder_10000_500.h5'))
# %%
"""
## loss curve
"""

# %%
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.savefig('enc_dec_loss')
plt.close()




  



