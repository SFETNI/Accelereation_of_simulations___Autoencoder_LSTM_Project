#!/bin/bash
# Submission script for Hercules
#SBATCH --partition=batch,hmem

# Lines beginning with #SBATCH are special commands to configure the job.

### Job Configuration Starts Here #############################################



# Export all current environment variables to the job (Don't change this)
#SBATCH --get-user-env
#SBATCH --output=slurm-job.%j.out   

# Submission script for Hercules
#SBATCH --time=12-00:00:00 # days-hh:mm:ss
#
#SBATCH --nodes=1
#Memory limit per compute node for the  job.  Do not use with mem-per-cpu flag.
#SBATCH --mem=256GB





#SBATCH --mail-user=s*****i@uliege.be
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

ulimit -s unlimited 

# activate virtual environnement

#virtualenv --system-site-packages ~/my_venv_pip
#source ~/my_venv_pip/bin/activate


echo "Job starting at $(date)" 

### Commands to run your program start here #################################### 
# --show-leak-kinds=definite --log-file=valgrind-output python  test.py  
#PYTHONMALLOC=malloc valgrind --leak-check=full --log-file=valgrind-output python  test.py  


# Note that we do not pass a redis password. The number of cpus needs to match nnodes*cpus-per-task

python data_to_txt.py  
#python load_data.py
echo "Job end at $(date)"  
