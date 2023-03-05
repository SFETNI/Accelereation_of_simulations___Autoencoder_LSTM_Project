#!/bin/bash
# Submission script for Nic5 
#SBATCH --partition=batch,hmem

# Lines beginning with #SBATCH are special commands to configure the job.

### Job Configuration Starts Here #############################################



# Export all current environment variables to the job (Don't change this)
#SBATCH --get-user-env
#SBATCH --output=slurm-job.%j.out   

# Submission script for Nic5
#SBATCH --time=2-00:00:00 # days-hh:mm:ss
#
#SBATCH --nodes=1
#Memory limit per compute node for the  job.  Do not use with mem-per-cpu flag.
#SBATCH --mem=512GB





#SBATCH --mail-user=*****@uliege.be
#SBATCH --mail-type=ALL



RUNPATH=$PWD
cd $RUNPATH

# force purge av modules (to avoid possible conflicts)
module --force purge

# load appropriate modules  (NIC5)
#module load releases/2021a
module load releases/2020b

module load Python

#module load Python/3.9.5-GCCcore-10.3.0


# activate virtual environnement

virtualenv --system-site-packages ~/my_venv_pip
source ~/my_venv_pip/bin/activate



echo "Job start at $(date)" 

### Commands to run your program start here ####################################
python encoder_decoder_LSTM_1_enc_dec_layer_PCA.py

echo "Job end at $(date)" 
