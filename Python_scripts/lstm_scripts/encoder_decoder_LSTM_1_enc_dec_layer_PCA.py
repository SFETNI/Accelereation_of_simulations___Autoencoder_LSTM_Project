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



    show_image(reco)

    plt.axis('off')

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
"""
# Import data
"""

# %%
dataset=np.load('/CECI/trsf/tmp_sf/dataset_7000.npy')

dataset=dataset[:1000]
# %%
"""
# Save all-files names in a list
"""




# %%
#list_names=np.loadtxt ('list_names_4800.txt', dtype=str)  

# %%
"""
## prepare data for encoder-decoder model
"""

# %%
encod_dataset=dataset.reshape(np.prod(dataset.shape[:2]),*dataset.shape[2:])

print(dataset.shape,encod_dataset.shape)

X=encod_dataset

# %%
# to erase: just to re

#X=X[:4800]

# %%
X = X.astype('float32') / 255.0 - 0.5

# %%
print('X.max(), X.min(): ',X.max(), X.min())

# %%
"""
## show sample(s)
"""

# %%
import matplotlib.pyplot as plt

def show_image(img):

    plt.imshow(np.clip(img + 0.5, 0, 1))
   

# %%
show_image(X[np.random.choice(range(len(X)), size=1)[0]])
plt.savefig('Original_Dataset_example')
plt.close()

# %%
from sklearn.model_selection import train_test_split

# the use of random_state is to reproduce the same results at each time 

X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

# %%
print('X_train.shape',X_train.shape)

print('X_test.shape)',X_test.shape)



# %%

# %%
"""
# load the encoded-decoded model
"""

# %%
from keras.models import load_model

save_dir='/home/sfetni/LSTM/saved_models/' #os.path.join(os.getcwd(), 'saved_computed_models')

 

#encoder_decoder = load_model(save_dir+model_name)

encoder= load_model(os.path.join(save_dir, 'encoder_7000_750.h5')) 
decoder= load_model(os.path.join(save_dir, 'decoder_7000_750.h5')) 


# %%
"""
## make predictions
"""

# %%
def visualize(img,encoder,decoder):

    """Draws original, encoded and decoded images"""

    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input

   

    code = encoder.predict(img[None])[0]

    reco = decoder.predict(code[None])[0]



    plt.subplot(1,3,1)

    plt.title("Original")

    show_image(img)



    plt.subplot(1,3,2)

    plt.title("Code")

    plt.imshow(code.reshape([code.shape[-1]//10,-1]))

    



    plt.subplot(1,3,3)

    plt.title("Reconstructed")

    show_image(reco)
    plt.show()
    



# plot 5 arbitrary frames and their images from the validation dataset

for i in range(5):

    img = X_test[np.random.choice(range(len(X_test)), size=1)[0]]

    visualize(img,encoder,decoder)
    plt.savefig('Original-code-Reconstructed')
    plt.close()






# %%
"""
# Retrieving compressed data from file
"""

# %%
loaded_arr = np.loadtxt("encoded_data_7000_750.txt")

code_size=750

# This loadedArr is a 2D array, therefore

# we need to convert it to the original

# array shape.reshaping to get original

# array with original shape.

load_original_arr = loaded_arr.reshape(

loaded_arr.shape[0], loaded_arr.shape[1] // code_size, code_size) # X_encoded.shape[2], X_encoded.shape[2])



X_encoded=load_original_arr

print(X_encoded.shape)

X_1=X_encoded

# %%
print(X_encoded.shape)

X_1=X_encoded.reshape(np.prod(X_encoded.shape[:2]),*X_encoded.shape[2:])

print(X_1.shape)

# %%
"""
## minor checks 
"""

# %%
code=X_1[np.random.choice(range(len(X_1)))]

reco = decoder.predict(code[None])[0]



show_image(denoise_image(denoise_image(reco)))

plt.title("Reconstructed")

plt.savefig('example-encoded_dataset')
plt.close()

print(reco.min(), reco.max())

# PCA 
print('PCA')
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
PowerTransformer.inverse_transform


def get_normalized_array(data,scaler):
    m=data.shape[0]
    data_scaled=[]
    for i in range(m):  
        x=data[i]
        x = np.round(x,20)
        x=x.reshape(len(x), 1)
        #x_scaled=scaler.transform(x)
        scaler=scaler.fit(x.reshape(len(x), 1))
        scaler_filename = "scaler"+str(i)+".save"
        joblib.dump(scaler, os.path.join("scalers", scaler_filename)) 
        x_scaled=scaler.transform(x)
        x_scaled=np.squeeze(x_scaled, axis=1)
        x_scaled = np.round(x_scaled,20)
        data_scaled.append(x_scaled)
    return np.asarray(data_scaled)


def get_rescaled_array(data):
    m=data.shape[0]
    data_rescaled=[]
    for i in range(m):  
        x=data[i]
        x = np.round(x,20)
        x = x.reshape(len(x), 1)
        scaler_filename = "scaler"+str(i)+".save"
        scaler = joblib.load(os.path.join("scalers", scaler_filename)) 
        x_rescaled=scaler.inverse_transform(x)
        x_rescaled=np.squeeze(x_rescaled, axis=1)
        x_rescaled = np.round(x_rescaled,20)
        data_rescaled.append(x_rescaled)
    return np.asarray(data_rescaled)
    
scaler= MinMaxScaler(feature_range=(0,1)) # StandardScaler() #

X_1_scaled=get_normalized_array(X_1[:],scaler) 

#X_1_recovered=get_rescaled_array(X_1_scaled,scaler)
print(X_1_scaled.min(),X_1_scaled.max())

X_1_rescaled=get_rescaled_array(X_1_scaled)

print(X_1[:].min(),X_1[:].max())
print(X_1_scaled.min(),X_1_scaled.max())
print(X_1_rescaled.min(),X_1_rescaled.max())

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(n_components=250 ) # 
X_1_pca=pca.fit_transform(X_1_scaled[:])  

print('X_1_pca.shape',X_1_pca.shape) 

plt.close()

plt.grid()
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.plot(np.cumsum(pca. explained_variance_ratio_ * 100))
plt.savefig('Explained variance')
plt.close()

X_1_pca_recov_scaled=pca.inverse_transform(X_1_pca)


X_1_pca_recov=get_rescaled_array(X_1_pca_recov_scaled)


print(X_1[:].shape, X_1[:].min(),X_1[:].max())
print(X_1_pca.shape,X_1_pca.min(),X_1_pca.max())
print(X_1_pca_recov.shape, X_1_pca_recov.min(),X_1_pca_recov.max())


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
"""
index_1= np.random.choice(range(len(X_encoded)), size=1)[0]
index_2= np.random.choice(range(X_encoded.shape[1]), size=1)[0]
index_1=3384 
index_2=90

print('(index 1,index 2): ',index_1,index_2)

code_1=X_1.reshape(X_encoded.shape)[index_1][index_2]
print('code_1.shape',code_1.shape)
# recode => original shape
img = decoder.predict(code_1[None])[0]

#plt.subplot(1,3,1)

plt.title("original")
plt.imshow(dataset[index_1][index_2])
plt.axis('off')
plt.show()
#plt.subplot(1,3,2)

show_image(denoise_image(denoise_image(img)))
plt.axis('off')
plt.title("Reconstructed -1 Layer")
plt.show()
print('original shape', img.shape)
print('code_1.shape', code_1.shape)
print('code_1.max',code_1.max())
print('X_1_pca.shape',X_1_pca.shape)



code_2=pca.transform(code_1.reshape(code_1.shape[0], 1))[1]
scaler_filename = "scaler"+str(index_2)+".save"
scaler = joblib.load(os.path.join("scalers", scaler_filename)) 
code_2=code_2.reshape(code_2.shape[0], 1)
code_2=scaler.inverse_transform(code_2)
code_2=np.squeeze(code_2, axis=1)

#code_2= X_1_pca.reshape(X_encoded.shape[0],X_encoded.shape[1] ,X_1_pca.shape[1])[index_1][index_2]  # code 2 will be used in LSTM 


#print('code_2.shape',code_2.shape)


#code_1_recov=pca.inverse_transform(code_2) # 

code_1_recov=X_1_pca_recov.reshape(X_encoded.shape)[index_1][index_2]

#print('code_1_recov.shape before rescale: ',code_1_recov.shape)

#print('code_1_recov.max() before rescale: ',code_1_recov.max())


#code_1_recov=code_1_recov.reshape(code_1_recov.shape[0], 1)
# scaler = scaler.fit(code_1_recov) 
#scaler=scaler.fit(X_1[-1].reshape(len(X_1[-1]), 1))
#code_1_recov=scaler.inverse_transform(code_1_recov)                           
#code_1_recov=np.squeeze(code_1_recov, axis=1)



print('code_1_recov.max() : ',code_1_recov.max())


img_1=decoder.predict(code_1_recov[None])[0]

#plt.subplot(1,3,3)

img_1_denoised=denoise_image(denoise_image(img_1))
show_image(img_1_denoised )
plt.axis('off')
plt.title("Reconstructed -2 Layers")
print('img_1.shape ',img_1.shape)


plt.savefig('Original-PCA-Reconstructed')
plt.close()
"""
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

## save X_1_pca in a text file as well as X_1_pca_recov
#np.savetxt('X_1_pca.txt',X_1_pca)

#X_1_pca=np.loadtxt('X_1_pca.txt')

"""
# LSTM model --------------------------------------------------------------------------------------------
"""

# %%
"""
## Prepare the new data : each image is represented by : 
### the main features of the simulation: concentration, mobililty , gradient coefficient
### the encoded array 
"""

# %%
import re
def get_features(str):
    array_str = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", str) 
    return np.array(array_str[:-1]).astype(float)


# %%
X_2=X_1_pca.reshape(X_encoded.shape[0],X_encoded.shape[1] ,X_1_pca.shape[1])
print(X_2.shape)

# %%
list_names=np.loadtxt ('list_names_7000.txt', dtype=str)  

# %%
X_new=[]
for i in range(X_2.shape[0]):    
    new_sequence=[]
    for j in range(X_2.shape[1]):
        features= get_features(list_names[i][j])
        concatenated_array=np.concatenate((features,X_2[i][j]), axis=0)
        new_sequence.append(concatenated_array) 
    X_new.append(new_sequence)
X_new=np.asarray(X_new)
print(X_new.shape)

# %%
"""
## Split Data into Train and Validation datasets
"""

# %%
# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.

def create_shifted_frames(data,N_output):
    x = data[:, 0 : data.shape[1] - N_output, :]   #data[:, 0 : data.shape[1] - 1, :]
    y =   data[:, data.shape[1]-N_output : data.shape[1], :]      #data[:, 1 : data.shape[1], :]
    return x, y

# %%
indexes = np.arange(X_new.shape[0])  #(
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * X_new.shape[0])] #X_encoded.shape[0]
val_index = indexes[int(0.9 * X_new.shape[0]) :]  
train_dataset = X_new[train_index]
val_dataset = X_new[val_index]

N_outputs=5   # predict the last N frames

# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset[:, 0:train_dataset.shape[1]:1],N_outputs)  #train_dataset.shape[1]:4
x_val, y_val = create_shifted_frames(val_dataset[:, 0:val_dataset.shape[1]:1],N_outputs)  #val_dataset.shape[1]:4

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))


# %%
"""
# Normalize data
"""

# %%
def get_normalized_data(array_data,scaler):
    
    x_train_scaled,y_train_scaled,x_val_scaled,y_val_scaled=[],[],[],[]
    dummy=[]
    for idx in range(len(array_data)):
        m=np.asarray(array_data[idx]).shape[0]
        n=np.asarray(array_data[idx]).shape[1]
        data=[]
        for i in range(m):  
            sequence=[] 
            for j in range(n):
                x=array_data[idx][i][j]
                x = x.reshape(len(x), 1)
                scaler = scaler.fit(x)
                x_scaled=scaler.transform(x)
                x_scaled=np.squeeze(x_scaled, axis=1)
                sequence.append(x_scaled)
            data.append(sequence)
        dummy.append(data)
    x_train_scaled,y_train_scaled,x_val_scaled,y_val_scaled=np.asarray(dummy[0]),np.asarray(dummy[1]),\
        np.asarray(dummy[2]),np.asarray(dummy[3])
    return  x_train_scaled,y_train_scaled,x_val_scaled,y_val_scaled


# %%
import numpy as np
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
scaler_ND = NDStandardScaler()
array_data=[x_train,y_train,x_val,y_val ]

x_train_scaled,y_train_scaled,x_val_scaled,y_val_scaled=get_normalized_data(array_data,scaler)
"""
print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
x_train_scaled = scaler_ND.fit_transform(x_train)
print(x_train_scaled.min(),x_train_scaled.max())
x_val_scaled = scaler_ND .transform(x_val)
y_train_scaled = scaler_ND.fit_transform(y_train)
print(y_train_scaled.min(),y_train_scaled.max())
y_val_scaled = scaler_ND.transform(y_val)
"""



# Re-Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train_scaled.shape) + ", " + str(y_train_scaled.shape))
print("Validation Dataset Shapes: " + str(x_val_scaled.shape) + ", " + str(y_val_scaled.shape))

# %%
"""
## minor checks
"""

# %%
i =np.random.choice(range(len(y_train_scaled)), size=1)[0]
j=np.random.choice(range(y_train_scaled.shape[1]), size=1)[0]
print('original:' ,'i= ',i,' j= ',j,' min: ',y_train[i][j].min(),' max: ',y_train[i][j].max())
print('scaled: ','i= ',i,' j= ',j,' min: ',y_train_scaled[i][j].min(),' max: ',y_train_scaled[i][j].max())

# %%
"""
## Model architecture
"""

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense  , RepeatVector, TimeDistributed
from keras.layers.core import Dense, Activation, Dense, Dropout

N_INPUTS=x_train_scaled.shape[1]
N_OUTPUTS=y_val_scaled.shape[1]
N_FEATURES=X_new.shape[2]
N_BLOCKS=1000


model = Sequential()
model.add(LSTM(N_BLOCKS, input_shape=(N_INPUTS, N_FEATURES) ))  
model.add(RepeatVector(N_OUTPUTS))

model.add(LSTM(N_BLOCKS, return_sequences=True)) 
model.add(LSTM(N_BLOCKS, return_sequences=True)) 

model.add(LSTM(N_BLOCKS, return_sequences=True)) 




model.add(layers.LSTM(N_BLOCKS, return_sequences=True))
"""
model.add(layers.Dropout(0.2))

model.add(layers.LSTM(N_BLOCKS, return_sequences=True))
model.add(layers.Dropout(0.2))

model.add(layers.LSTM(N_BLOCKS, return_sequences=True))
model.add(layers.Dropout(0.2))
"""

model.add(Dense(N_FEATURES))
#model.add(Activation('linear'))
model.compile(loss='mae', optimizer='adam')
model.summary()



# %%
from keras.callbacks import EarlyStopping 
earlyStop = EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=10)
model_checkpoint=tf.keras.callbacks.ModelCheckpoint('CIFAR10{epoch:02d}.h5',save_freq=2,save_weights_only=False)
history=model.fit(x_train_scaled, y_train_scaled,
          batch_size=5, epochs=50,
          validation_data=(x_val_scaled, y_val_scaled), callbacks=earlyStop)


# %%
"""
##show loss
"""

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('LSTM_loss')
plt.close()


# %%
"""
## save the model
"""

# %%
lstm_model_name='lstm_7000_750_pca_250.h5'
model.save(lstm_model_name)

# %%
"""
## load saved model
"""

# %%
model=load_model(lstm_model_name)
#model=load_model(os.path.join(lstm_model_name))

# %%
"""
# make predictions
"""

# %%
index= np.random.choice(range(len(x_val_scaled)), size=1)[0]
index=608
#example =x_val_scaled[index]
example =x_val_scaled[index] 
print('example.shape : ',example.shape)
actual_frames=example[:N_INPUTS, ...]
print('shape of frames to predict: ',actual_frames.shape)

new_prediction = model.predict(np.expand_dims(actual_frames, axis=0))
new_prediction = np.squeeze(new_prediction, axis=0)
print('new_prediction.shape: ',new_prediction.shape)

sequence = np.concatenate((actual_frames, new_prediction), axis=0)
print('"original_frames+prediction" shape ',sequence .shape)

print('actual_frames.shape',actual_frames.shape,actual_frames[0].shape)
print('min,max',actual_frames[0].min(),actual_frames[0].max())

print('new_prediction.shape',new_prediction.shape,new_prediction[0].shape)
print('min,max',new_prediction[0].min(),new_prediction[0].max())
new_prediction=new_prediction[::, 3:new_prediction.shape[1]]  # uncomment if characteristics are not considered
print('new_prediction.shape',new_prediction.shape)
"""
from sklearn.metrics import mean_squared_error, r2_score
index_1=np.random.choice(range(len(y_val_scaled[::, ::,3:y_val.shape[2]])), size=1)[0]
y_actual= y_val_scaled[::, ::,3:y_val.shape[2]][index_1]
y_predicted=new_prediction
rms = mean_squared_error(y_actual, y_predicted, squared=False)
r2=r2_score(y_actual[0], y_predicted[0])
print('r2 :', r2, ' RMSE: ',rms )
"""


# %%
"""
## plot predictions  (for small sequence)
"""

# %%
y_val=y_val[::, ::,3:y_val.shape[2]]
print(y_val.shape)
print(y_val.reshape(np.prod(y_val.shape[:2]),*y_val.shape[2:]).shape)

# %%
"""
#### y_val_rescaled ;  the shape of the code (auto-encoder)
"""

# %%
y_val_recovered=pca.inverse_transform(y_val.reshape(np.prod(y_val.shape[:2]),*y_val.shape[2:]))
print(y_val_recovered.shape)

# %%
# reshape y_val_rescaled
print(y_val.shape)
origninal_shape=(y_val.shape[0],y_val.shape[1],y_val_recovered.shape[-1])
print(y_val_recovered.min(),y_val_recovered.max())
y_val_recovered=get_rescaled_array(y_val_recovered).reshape(origninal_shape)
print(y_val_recovered.shape)


# %%
"""
#### new_prediction_rescaled => the shape of the code (auto-encoder)
"""

# %%
print(new_prediction.shape)
#new_prediction_recovered=get_rescaled_array(pca.inverse_transform(new_prediction))
new_prediction_recovered=pca.inverse_transform(new_prediction)
print(new_prediction_recovered.shape)

# %%
#encoded_original_frames=y_val[index]

#  original last n frames (to predict)
encoded_original_frames= y_val_recovered[index] #[::, ::,3:y_val.shape[2]][index]  #encoded => auto-encoder


# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 5, figsize=(20, 4))

# original frames
for idx, ax in enumerate(axes[0]):
    code=encoded_original_frames[idx]
    #print(code.shape)
    img=decoder.predict(code[None])[0]
    ax.imshow(np.clip(img + 0.5, 0, 1))
    ax.set_title(f"Frame {idx + N_INPUTS+1}")
    ax.axis("off")
    

# prediction
for idx, ax in enumerate(axes[1]):
    code=new_prediction_recovered[idx]
    # print(code.min(),code.max())
    #print(code.shape)
    #encoded_scaled_frame = code.reshape(encoded_scaled_frame.shape[0], 1)
    #frame_inverted=scaler.inverse_transform(encoded_scaled_frame)   # you should use the same
    #frame_inverted=np.squeeze(frame_inverted, axis=1)

    img=decoder.predict(code[None])[0]
    ax.imshow(np.clip(img+0.5, 0, 1))
    #ax.imshow(frame)
    ax.set_title(f"Frame {idx + N_INPUTS+1}") 
    ax.axis("off")
plt.savefig('predictions')