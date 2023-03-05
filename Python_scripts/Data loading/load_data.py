

# %%
import numpy as np
import matplotlib.pyplot as plt
import time 
import math
import pyvista as pv
import pyfftw          # use for fast fourier transform
from scipy.fft import fft, ifft
from numba import jit  # use to speed up 
import scipy.stats as st
import time
from scipy.sparse import csgraph
import shutil
import os
import ntpath







import os



dataset_loaded=np.load('dataset_12.npy')

print(dataset_loaded.shape)

def show_image(x):
    plt.matshow(x,cmap='viridis')
    print('ok')
    plt.savefig('img.png')
    plt.close()



print(dataset_loaded[-1][-1].shape)
show_image(dataset_loaded[-1][-1])
