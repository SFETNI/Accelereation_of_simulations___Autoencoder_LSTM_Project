# %%
"""
# Import
"""

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


# %%
"""
# Functions
"""

# %%
#--------------------------------------------------------------------------------------------------
def free_energ(c):

    A=1.0

    dfdc =A*(2.0*c*(1-c)**2 -2.0*c**2 *(1.0-c))

    return dfdc



#--------------------------------------------------------------------------------------------------
def fft_(a):

    """

    return a fft object from pyfftw library that will be use to compute fft

    """

    fft_object=pyfftw.builders.fftn(a,axes=(0,1,2), threads=12)

    return fft_object()

#--------------------------------------------------------------------------------------------------

def ifft_(a):

    """

    return a inverse fft object from pyfftw library that will be use to compute inverse fft

    """

    ifft_object=pyfftw.builders.ifftn(a,axes=(0,1,2), threads=12)

    return ifft_object()

#--------------------------------------------------------------------------------------------------

#@jit(nopython=True)

def micro_ch_pre(Nx,Ny,Nz,c0):

    c=np.zeros((Nx,Ny,Nz))

    noise=0.01

    for i_x in range(Nx):

        for i_y in range(Ny):

            for i_z in range(Nz):

                c[i_x,i_y,i_z] =c0 + noise*(0.5-np.random.rand())                 

    return c


# Compute energy evolution

def calculate_energ(Nx,Ny,Nz,c,grad_coef):

    energ =0.0

    # ------------Nx------------------------------

    for i in range (Nx-1): 

        ip = i + 1

        #--------------Ny-------------------------

        if (Ny>1):

            for j in range (Ny-1):

                jp = j + 1

                # ----------Nz--------------------

                if (Nz>1): # 3D

                    for l in range (Nz-1):

                        lp = l + 1 

                        energ += c[i,j,l]**2  *(1.0-c[i,j,l])**2  +  0.5*grad_coef*((c[ip,j,l]-c[i,j,l])**2 \

                            +(c[i,jp,l]-c[i,j,l])**2     + (c[i,j,lp]-c[i,j,l])**2)

                else: # (Nz==1) 2D

                        energ += c[i,j,0]**2  *(1.0-c[i,j,0])**2  +  0.5*grad_coef*((c[ip,j,0]-c[i,j,0])**2 \

                        +(c[i,jp,0]-c[i,j,0])**2) 

        else : # (Ny==1): 

            # ----------Nz--------------------

            if (Nz>1): # 2D

                 for l in range (Nz-1):

                     lp = l + 1

                     energ += c[i,0,l]**2  *(1.0-c[i,0,l])**2  +  0.5*grad_coef*((c[ip,0,l]-c[i,0,l])**2 \

                     + (c[i,0,lp]-c[i,0,l])**2)       

            else : # (Nz==1) 1D

                energ += c[i,0,0]**2  *(1.0-c[i,0,0])**2  +  0.5*grad_coef*(c[ip,0,0]-c[i,0,0])**2                         

         

    return energ


# plot micro ------------------------------------------------------------

def plot_micro(c,opt,ttime,child_path,index_save):

    # 1D case

    if (opt=='1D'):

        plt.plot(c[:, :, 0])

        plt.xlabel('x')

        plt.ylabel('concentration')

        plt.title('initial concentration')

    else: 

        # 2D or 3D cases--------------------------------------------------

        import sys
        
        grid  = pv.UniformGrid()

        grid.spacing=np.array([dx,dx,dx])*1E9

        grid.dimensions = np.array([Nx,Ny,Nz])#+1

        grid.point_arrays[r'c'] = np.transpose(np.resize(c,[Nx,Ny,Nz])).flatten()


        # Set a custom position and size

        sargs = dict(fmt="%.1f", color='black') 
        pv.set_plot_theme("ParaView")

        p = pv.Plotter(window_size=[500, 500],off_screen=True)

        p.set_background("white")

        p.add_mesh(grid,show_scalar_bar=False,label='title')
        #p.add_scalar_bar('Concentration', color='black',label_font_size=12, width=0.1, height=0.7, position_x=1,position_y=0.16,vertical= True, interactive=False)
        
        """

        if (ttime==0):

            p.add_text('Initial Microstructure ',position='upper_edge',color='black',font= 'times',font_size=12)

        else:

            p.add_text('Microstructure at dimensionless time '+str('{0:.2f}'.format(ttime) ),color='black',font= 'times',font_size=12)

        """

        
        #p.show_bounds(all_edges=True,font_size=24,bold=True, xlabel="X [nm]",ylabel="Y [nm]",color='black')

        #p.add_title('Microstructure at dimensionless time '+str('{0:.2f}'.format(ttime) ))

        p.camera_position = 'xy'

        if (opt=='2D'):
           filename = ntpath.basename(child_path)+'___t_'+str(index_save)+'.png'
           #p.screenshot(os.path.join(child_path, filename))
           im1=plt.matshow(c,cmap='viridis')
           plt.axis('off')
           #plt.colorbar(im1)
           plt.savefig(os.path.join(child_path, filename),bbox_inches="tight",pad_inches=0,dpi=1200)
           plt.close()

        elif (opt=='3D'):

            p.show(screenshot=child_path+os.path.basename(os.path.dirname(child_path))+'___t_'+str(index_save)+'.png',cpos="xy") 

        p.close()

# -------------------------------------------------------------------------------

def save_micro_in_file(c,ttime,child_path,index_save):

    #--- save micro in file
    filename = ntpath.basename(child_path)+'___t_'+str(index_save)+'.txt'
    a_file = open(os.path.join(child_path, filename), "w")
    for row in c[:, :, 0]:
        np.savetxt(a_file, row)
    a_file.close()



    a_file.close()

# -------------------------------------------------------------------------------

# ------------ erase folder---------------------------------------------------- 

def EraseFile(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# %%
@jit(nopython=True)
def prepar_fft(Nx,dx,Ny,dy,Nz,dz,opt): 
    """
    Compute spatial frequence term and derivative
    """
    # variable initialisation
    lin_x=np.zeros(Nx)
    lin_y=np.zeros(Ny)
    lin_z=np.zeros(Nz)
        
    k=np.zeros((3,Nx,Ny,Nz))
    k2=np.zeros((Nx,Ny,Nz))
    k4=np.zeros((Nx,Ny,Nz))
        
    """
    # Method 1 to compute k  (3D)
    if (Nx % 2) == 1 : # = number odd if remainers is one
        lin_x[:int((Nx-1)/2.0+1)]=np.arange(0, int((Nx-1)/2.0+1), 1)*2*np.pi/(Nx*dx)
        lin_x[int((Nx-1)/2.0+1):]=np.arange(int(-(Nx+1)/2.0 +1), 0, 1)*2*np.pi/(Nx*dx)
    if (Ny % 2) == 1 :
        lin_y[:int((Ny-1)/2.0+1)]=np.arange(0, int((Ny-1)/2.0+1), 1)*2*np.pi/(Ny*dy)
        lin_y[int((Ny-1)/2.0+1):]=np.arange(int(-(Ny+1)/2.0 +1), 0, 1)*2*np.pi/(Ny*dy)
    if (Nz % 2) == 1 :
        lin_z[:int((Nz-1)/2.0+1)]=np.arange(0, int((Nz-1)/2.0+1), 1)*2*np.pi/(Nz*dz)
        lin_z[int((Nz-1)/2.0+1):]=np.arange(int(-(Nz+1)/2.0 +1), 0, 1)*2*np.pi/(Nz*dz)        
    if (Nx % 2) == 0 : # = number even if remainers is zero
        lin_x[0:int(Nx/2.0)]=np.arange(0, int(Nx/2.0), 1)*2*np.pi/(Nx*dx)
        lin_x[int(Nx/2.0 + 1):]=np.arange(int(-Nx/2.0 + 1), 0, 1)*2*np.pi/(Nx*dx)
    if (Ny % 2) == 0 :
        lin_y[0:int(Ny/2.0)]=np.arange(0, int(Ny/2.0), 1)*2*np.pi/(Ny*dy)
        lin_y[int(Ny/2.0 + 1):]=np.arange(int(-Ny/2.0 + 1), 0, 1)*2*np.pi/(Ny*dy)  
    if (Nz % 2) == 0 :
        lin_z[0:int(Nz/2.0)]=np.arange(0, int(Nz/2.0), 1)*2*np.pi/(Nz*dz)
        lin_z[int(Nz/2.0 + 1):]=np.arange(int(-Nz/2.0 + 1), 0, 1)*2*np.pi/(Nz*dz)          
    # grid   
    for i in range(Nx):
        for j in range(Ny):
            for l in range(Nz):
                 k[0,i,j,l]= lin_x[i]
                 k[1,i,j,l]= lin_y[j]
                 k[2,i,j,l]= lin_z[l]
     """
    # Method 2 to compute k      
    Lx=Nx*dx
    x=np.linspace(-0.5*Lx+dx,0.5*Lx,Nx)
    
    Ly=Ny*dy
    y=np.linspace(-0.5*Ly+dy,0.5*Ly,Ny)
    
    Lz=Nz*dz
    z=np.linspace(-0.5*Lz+dz,0.5*Lz,Nz)
         
    
    xx=2*np.pi/Lx*np.concatenate((np.arange(0,Nx/2+1), np.arange(-Nx/2+1,0)),axis=0)
    yy=2*np.pi/Ly*np.concatenate((np.arange(0,Ny/2+1), np.arange(-Ny/2+1,0)),axis=0)
    zz=2*np.pi/Lz*np.concatenate((np.arange(0,Nz/2+1), np.arange(-Nz/2+1,0)),axis=0)
        
    for i in range(Nx):
        for j in range(Ny):
            for l in range(Nz):
                k[0,i,j,l]= xx[i]
                k[1,i,j,l]= yy[j]
                k[2,i,j,l]= zz[l]           
                
    k2=k[0]**2+k[1]**2+k[2]**2
  
    k4=k2**2
         
    return k,k2,k4,x,y,z

# %%
"""
# Input Data
"""

# %%
Nx=128 ; Ny=128; Nz=1
# spacing
dx=10e-10 # [m]
dy=10e-10 # [m]
dz=10e-10 # [m]

# convert into adimensional grid
dx_s=dx/dx
dy_s=dy/dy
dz_s=dz/dz


# %%
# Choose the time steps to be varied
#array_time_steps=np.arange(0.1,1,0.1) # uncomment if you try many time steps : useful in trial and error approach
array_time_steps=[0.1] # if you choose one time step
array_dtime=[]

# %%
"""
c0=0.4  #initial microstructure 
mobility =1.0
coefA = 1.0
grad_coef=0.5

#initialize microstructure
c= micro_ch_pre(Nx,Ny,Nz,c0)
c0_save=c
"""
"""
#uncomment if necessary
# For comparison purposes: load the same initial microtructure as another reference (Matlab- Biner 2017 for example)
# retrieving data from file.
loaded_arr = np.loadtxt("3D_micro_init.txt")

# This loadedArr is a 2D array, therefore
# we need to convert it to the original
# array shape.reshaping to get original
# matrice with original shape.
load_original_arr = loaded_arr.reshape(
loaded_arr.shape[0], loaded_arr.shape[1] // c.shape[2], c.shape[2])

c=load_original_arr
"""

# %%
"""
## save c in a text file : for comparison purposes
"""

# %%
"""
a_file = open("c0_save.txt", "w")
for row in c[:, :, 0]:
    np.savetxt(a_file, row)

a_file.close()
"""

# %%
"""
original_c = np.loadtxt("c0_save.txt").reshape(Nx,Ny)
c[:, :, 0]=original_c
"""

# %%
"""
# plot initial microstructure
"""

# %%
parent_path= os.path.join(os.getcwd(),"sim_data")
if not os.path.exists(parent_path):
    os.makedirs(parent_path)

print(parent_path)
# %%
"""
ttime=0
plot_micro(c,"2D",ttime,parent_path,0)  
""" 

# %%
"""
# Generate Data Base

"""
# %%
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------

EraseFile(parent_path)
print('done')
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------

# start simulation

t_start = time.time() 

#--------------------------------------------------------------------------------------------

array_k=np.arange(0.35,0.85,0.1)

for index_k in np.arange(len(array_k)):   # to change gradient coefficient

    array_mobility= np.arange(1,2.2,0.1)

    for index_m in range (len(array_mobility)):   # to change the mobility 

        array_c_mean=np.arange(0.55,0.76,0.01)

        for index_c_init in range (len(array_c_mean)):   # to change the mean concentration, mobility or gradient coefficient

            c0=array_c_mean[index_c_init]  #initial microstructure 

            mobility =array_mobility[index_m] 

            coefA = 1.0

            grad_coef=array_k[index_k]   # gradient coefficient

            #initialize microstructure

            c= micro_ch_pre(Nx,Ny,Nz,c0)

            c0_save=c



            index_save=1   # to save images and/or matrix for each sequance : t1, t2 ... tn

            # loop to change time step (dtime) and compute associated energy dissipation for CH equation (at each time step) 

            for index in range(len(array_time_steps)):   

                # create a new child directory (if not exist) to store data for each sequence 

                child_path=os.path.join(parent_path,"sequence_c="+"% s" % '{0:.3f}'.format(c0)+"_M="+"% s" % '{0:.2f}'.format(mobility)+"_k="+"% s" % '{0:.2f}'.format(grad_coef))

                if not os.path.exists(child_path):

                       os.makedirs(child_path)



                c=c0_save  # take the same microstructure

                # time step and constant values

                ttime=0 # for each simulation 

                dtime=array_time_steps[index]

                #------------------

                # time steps and print parameters

                Nt=1000 # trial

                #nstep=14450 #

                nstep=int(round(Nt/dtime)) #

                nprint=100 # step to print : print at each time step (istep)

                # set fourier coefficient 

                # compute the spatial frequency term from fft

                k,k2,k4,x,y,z=prepar_fft(Nx,dx_s,Ny,dy_s,Nz,dz_s,opt="3d")

                dfdc=np.zeros((Nx,Ny,Nz))

                array_energy=[]

                array_time=np.zeros(nstep)  # to save time steps

                array_c=[]

                dfdc=np.zeros((Nx,Ny,Nz))



                



                endstep=nstep

                flag=0

                #-------------------------------------------------

                for istep in range(nstep):

                #-------------------------------------------------

                    if (flag==1):

                        endstep=istep

                        break #break the loop file

                    



                    ttime = ttime + dtime  

                    array_dtime.append(dtime)

                    # compute free energy 

                    dfdc=free_energ(c)

                        

                    dfdck=fft_(dfdc) 

                    ck=fft_(c)



                    # Time integration

                    numer=dtime*mobility*k2*dfdck

                    denom = 1.0 + dtime*coefA*mobility*grad_coef*k4

                    ck =(ck-numer)/denom

                    

                    c=np.real(ifft_(ck))   



                    # for small deviations

                    c[np.where(c >= 0.9999)]= 0.9999

                    c[np.where(c <= 0.00001)]=0.00001  



                    energy=calculate_energ(Nx,Ny,Nz,c,grad_coef)

                    array_energy.append(energy)

                    array_time[istep]=ttime

                    array_c.append(c)

                    

                    energ = np.array(array_energy)



                    # uncomment if you define a stop criteria for your simulation

                    """

                    # ---------------------------

                    if (energ[istep]<50):



                        plot_micro(c,"2D",ttime)

                        #--- save micro in file

                        a_file = open("c_"+str('{0:.0f}'.format(int(ttime)))+".txt", "w")

                        for row in c[:, :, 0]:

                            np.savetxt(a_file, row)



                        a_file.close()

                        print('break simulation')

                        flag=1

                    #---------------------------

                    """

                    if (math.fmod(istep,nprint)==0):

                        plot_micro(c,"2D",ttime,child_path,index_save)

                        #save_micro_in_file(c,ttime,child_path,index_save)

                        index_save+=1

                    """" 

                    if ((math.fmod(int(energ),20)==0):

                        plot_micro(c,"2D",ttime)

                    """    


            

CPU=time.time() - t_start            

print("simulation time : %s seconds ---" % (time.time() - t_start))


# %%
