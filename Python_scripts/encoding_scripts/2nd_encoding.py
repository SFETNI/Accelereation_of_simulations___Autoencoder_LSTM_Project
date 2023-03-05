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
from skimage.color import rgb2gray

from keras.callbacks import EarlyStopping
import os
from sklearn.preprocessing import MinMaxScaler
#from numba import jit  # use to speed up python

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
#@jit
def load_images_from_folder(folder):
    images = []
    list=sorted_alphanumeric(os.listdir(folder))

    for filename in list:

        img = cv2.imread(os.path.join(folder,filename))  # if binary, cv2.imread(os.path.join(folder,filename),2)
        img=cv2.resize(img, (128,128))

        #If desired binary images in output
        #img=img[:,:,0]                #  uncomment (these 2 lines) if the desired output is  binary images (method 1 ti make images binary )
        #img=img.reshape(128,128,1)    #  idm

        #img=rgb2gray(img)   #  method 2 ti make images binary 
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
#@jit
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
#@jit
def load_all_data(input_dir):
    data_shape = get_data_shape(input_dir) 
    #np.empty(shape=(data_shape[0],data_shape[1],data_shape[2],data_shape[3],data_shape[4] ))  
    #data=np.empty([data_shape[1],data_shape[2],data_shape[3],data_shape[4] ])  
    data=[]
    folders = os.listdir(input_dir)
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
def invert_transform(data,scaler):
    for i in range(data.shape[0]):   
        for j in range(data.shape[1]):
            x=data[i][j]
            x = x.reshape(len(x), 1)
            scaler = scaler.fit(x)  
            x_inverted=scaler.inverse_transform(x)
            x_inverted=np.squeeze(x_inverted, axis=1)
            data[i][j]=x_inverted
    return data

# %%
def denoise_image(reco):
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
   
    plt.axis('off')
    show_image(reco)
    
    plt.savefig('reco',bbox_inches="tight",pad_inches=0)
    img= cv2.imread('reco.png')
    plt.close()
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 10, 7)

    dst =cv2.resize(dst,(128,128),interpolation= cv2.INTER_LINEAR)
    """
    # Plotting of source and destination image
    plt.subplot(121), show_image(img.astype('float32') / 255.0 - 0.5)
    plt.title("Before")

    plt.subplot(122), show_image(dst.astype('float32') / 255.0 - 0.5)
    plt.title("After")
    """
    reco_denoised=dst.astype('float32') / 255.0 - 0.5

    #print(reco_denoised.shape)
    #print(reco.min(), reco.max())
    #print(reco_denoised.min(), reco_denoised.max())
    return  reco_denoised

# %%

# %%
"""
## show sample(s)
"""

# %%
import matplotlib.pyplot as plt
def show_image(img):
    plt.imshow(np.clip(img + 0.5, 0, 1))  # uncomment if RGB
    #plt.imshow(img) # if binary





# %%
"""
# load the encoded-decoded model
"""

# %%
from keras.models import load_model
save_dir='/workdir/sfetni/LSTM/train_LSTM'
import os 
#encoder_decoder = load_model(save_dir+model_name)
encoder= load_model(os.path.join(save_dir, 'encoder_7000_750.h5'))  
decoder= load_model(os.path.join(save_dir, 'decoder_7000_750.h5'))   




# %%
"""
# Prepare the encoded Dataset 
"""

# %%
# remind that X is the dataset (normalized :/255 and centred :-0.5)
"""
X=X.reshape(dataset.shape)
encoded_data=[]
for i in range(dataset.shape[0]):   
    encoded_sequence=[]
    for j in range(dataset.shape[1]):
        img=X[i][j]
        encoded_frame= encoder.predict(img[None])[0]
        encoded_sequence.append(encoded_frame)

    encoded_data.append(encoded_sequence)
X_encoded=np.asarray(encoded_data)
print(X_encoded.shape)
"""


# %%
"""
# Retrieving compressed data from file
"""

# %%
loaded_arr = np.loadtxt(os.path.join(save_dir, "encoded_data_7000_750.txt"))  
#loaded_arr = np.loadtxt("encoded_data_10000_1000.txt")
code_size=750
# This loadedArr is a 2D array, therefore
# we need to convert it to the original
# array shape .reshaping to get original
# array with original shape.        

load_original_arr = loaded_arr.reshape(
loaded_arr.shape[0], loaded_arr.shape[1] // code_size, code_size) # X_encoded.shape[2], X_encoded.shape[2])

X_encoded=load_original_arr
print(X_encoded.shape)
X_1=X_encoded 

# %%
"""
### To retrieve supplementary data
"""

# %%
loaded_arr_supp = np.loadtxt(os.path.join(save_dir, "encoded_data_1700_sup_750.txt"))  
load_original_arr_supp = loaded_arr_supp.reshape(
loaded_arr_supp .shape[0], loaded_arr_supp .shape[1] // code_size, code_size) # X_encoded.shape[2], X_encoded.shape[2])

X_encoded_supp=load_original_arr_supp 
print(X_encoded_supp .shape)
X_1_supp=X_encoded_supp

# %%
"""
X_encoded_denoised=[]
for i in range(X_encoded.shape[0]):    
    encoded_sequence=[]
    for j in range(X_encoded.shape[1]):
        code=X_encoded[i][j]
        reco = decoder.predict(code[None])[0]
        encoded_frame= denoise_image(denoise_image(reco))
        encoded_sequence.append(encoded_frame)

    X_encoded_denoised.append(encoded_sequence)
X_encoded_denoised=np.asarray(X_encoded_denoised)
print(X_encoded_denoised.shape)
"""

# %%
"""
#### save X_encoded 
"""

# %%
X_encoded_save=X_encoded[:]
X_encoded_save=np.asarray(list(X_encoded))
print(X_encoded_save.shape)

# %%
X_1=X_encoded_save  # if X_1 is hereafter modified and we need to restart from previous cell

# %%
"""
#### reduced dataset (to be more homogenous)
"""

# %%
import re
def get_features(str):
    array_str = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", str) 
    return np.array(array_str[:-1]).astype(float)


# %%
list_names= np.loadtxt(os.path.join(save_dir, 'list_names_7000.txt'), dtype=str)
print(list_names.shape)

# %%
# only if there is supplementary data
list_names_supp= np.loadtxt(os.path.join(save_dir, 'list_names_sup_1700.txt'), dtype=str)
print(list_names_supp.shape)

# %%
"""
#### This function is to select some data from X_1
"""

# %%
#from numba import jit  # use to speed up python
def reduced_dataset(X_encoded,list_names):
    X_encoded_reduced=[]
    list_names_reduced=[]
    for i in range(X_encoded.shape[0]):    #
        encoded_sequence=[]
        features= get_features(list_names[i][0])
        c,m,k=features[0],features[1],features[2]
        #print(c,m,k)
        if (c<=0.7) and (m>=0.8) and (m<=1.2) and (k<=0.5):
           X_encoded_reduced.append(X_encoded[i])
           list_names_reduced.append(list_names[i])
    X_encoded_reduced=np.asarray(X_encoded_reduced)
    list_names_reduced=np.asarray(list_names_reduced)

    return  X_encoded_reduced,list_names_reduced

# %%
"""
### if you select some data from X_1
"""

# %%
"""
X_1,list_names_reduced=reduced_dataset(X_encoded,list_names)
print(X_1.shape)
print(list_names_reduced.shape)
X_encoded_reduced=X_1[:]
X_encoded_reduced=np.asarray(list(X_1))
"""

# %%
"""
#### X1 => 3D to 2D transform (Reshape)  => Mandatory for PCA decomposition
"""

# %%
#X_encoded=X_encoded[:3500]
#X_encoded=X_encoded_save
#print(X_encoded.shape)
N=X_1.shape[0] # 3000 #X_encoded.shape[0]
#X_1=X_encoded[:N].reshape(np.prod(X_encoded[:N].shape[:2]),*X_encoded[:N].shape[2:])
X_1=X_1[:N].reshape(np.prod(X_1[:N].shape[:2]),*X_1[:N].shape[2:])
print(X_1.shape)  # X_1: to 2nd layer of encoder or PCA reduction 
#print(X_encoded_reduced.shape)

# %%
"""
#### reshape supplementary data (if there is)
"""

# %%
X_1_supp=X_1_supp.reshape(np.prod(X_1_supp.shape[:2]),*X_1_supp.shape[2:])
print(X_1_supp.shape)

# %%
X_1_new=np.concatenate([X_1,X_1_supp],axis=0)
print(X_1_new.shape)

# %%
# save X_1_new in txt file  
#np.savetxt("X_1_new.txt",X_1_new)

# %%
#X_1_new=np.loadtxt("X_1_new.txt")  

# %%
print(X_1_new.shape)

# %%
"""
## minor checks 
"""

# %%
code=X_1_new[np.random.choice(range(len(X_1_new)))]
print(code.max())
reco = decoder.predict(code[None])[0]

show_image(denoise_image(denoise_image(reco)))
plt.title("Reconstructed")
plt.savefig("Reconstructed")
plt.close()
print(reco.min(), reco.max())



# %%
"""
# Layer 2 ------------------ Encoder Decoder -------------------------
# ------------------------------------------------------------------------
# if you use a second layer of autoencoder ; else go directly to the section PCA reduction
"""

# %%
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = MinMaxScaler(feature_range=(0,1)) # StandardScaler() #
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def inverse_transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.inverse_transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

# %%
from sklearn.model_selection import train_test_split
# the use of random_state is to reproduce the same results at each time 
X_1_train, X_1_test = train_test_split(X_1_new, test_size=0.1, random_state=42)
print(X_1_train.shape)
print(X_1_test.shape)


scaler= MinMaxScaler(feature_range=(0,1))



X_1_train_scaled=scaler.fit_transform(X_1_train)

X_1_test_scaled=scaler.transform(X_1_test)


# Re-Inspect the dataset.
print("Training Dataset Shapes: " + str(X_1_train_scaled.shape) + ", " + str(X_1_train_scaled.shape))
print("Validation Dataset Shapes: " + str(X_1_test_scaled.shape) + ", " + str(X_1_test_scaled.shape))

# %%
#X_1_train=X_1_train.reshape(np.prod(X_1_train.shape[:2]),*X_1_train.shape[2:])
#X_1_test = X_1_test.reshape(np.prod(X_1_test.shape[:2]),*X_1_test.shape[2:])

# %%
# note that X_11 is the reshaped version of X_1
"""
X_11_train=X_1_train.reshape(np.prod(X_1_train.shape[:2]),*X_1_train.shape[2:])
X_11_test = X_1_test.reshape(np.prod(X_1_test.shape[:2]),*X_1_test.shape[2:])
X_11_train_scaled =X_1_train_scaled.reshape(np.prod(X_1_train_scaled.shape[:2]),*X_1_train_scaled.shape[2:])
X_11_test_scaled = X_1_test_scaled.reshape(np.prod(X_1_test_scaled.shape[:2]),*X_1_test_scaled.shape[2:])
"""

# %%
i =np.random.choice(range(len(X_1_train_scaled)), size=1)[0]
print('original:' ,'i= ',i, X_1_train[i].min(),X_1_train[i].max())
print('scaled: ','i= ',i, X_1_train_scaled[i].min(),X_1_train_scaled[i].max())

# %%
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

# %%
def build_autoencoder_2(img_shape, code_size):
    # The encoder
    encoder = Sequential()  
    encoder.add(InputLayer(img_shape))
    encoder.add(Dense(700))
    encoder.add(Dense(650))
    encoder.add(Dense(600))
    encoder.add(Dense(550))
    encoder.add(Dense(500))
    encoder.add(Dense(450))
    encoder.add(Dense(400))
    encoder.add(Dense(350))
    encoder.add(Dense(300))
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(Dense(code_size))
    decoder.add(Dense(300))
    decoder.add(Dense(350))
    decoder.add(Dense(400))
    decoder.add(Dense(450))
    decoder.add(Dense(500))
    decoder.add(Dense(550))
    decoder.add(Dense(600))
    decoder.add(Dense(650))
    decoder.add(Dense(700))
    decoder.add(Dense(np.prod(img_shape))) 
    decoder.add(Reshape(img_shape))

    return encoder, decoder

# %%
Code_SHAPE =X_1.shape[1:] 

encoder_2, decoder_2 = build_autoencoder_2(Code_SHAPE, 250)

inp = Input(Code_SHAPE)
code = encoder_2(inp)
reconstruction = decoder_2(code)

autoencoder_2 = Model(inp,reconstruction)
autoencoder_2.compile(optimizer= 'adamax', loss='mse', metrics=['MSE'])  #binary_crossentropy
callbacks = [EarlyStopping(monitor='MSE', min_delta=0, patience=10)]
print(autoencoder_2.summary())

# %%
history = autoencoder_2.fit(X_1_train_scaled, X_1_train_scaled,
                epochs=15,
                batch_size=64, 
                shuffle=False,
                validation_data=(X_1_test_scaled, X_1_test_scaled))  #callbacks=[model_checkpoint]
                
                
                
encoder_2.save('encoder_7000_750_250.h5')

decoder_2.save('decoder_7000_750_250.h5')                

