#!/bin/bash
# Submission script for Nic5 
#SBATCH --partition=batch

# Lines beginning with #SBATCH are special commands to configure the job.

### Job Configuration Starts Here #############################################



# Export all current environment variables to the job (Don't change this)
#SBATCH --get-user-env
#SBATCH --output=slurm-job.%j.out   

# Submission script for Nic5
#SBATCH --time=6-00:00:00 # days-hh:mm:ss
#
#SBATCH --nodes=1
#Memory limit per compute node for the  job.  Do not use with mem-per-cpu flag.
#SBATCH --mem=64GB




#SBATCH --mail-user=*****@uliege.be
#SBATCH --mail-type=ALL




#SBATCH --requeue

RUNPATH=$PWD
cd $RUNPATH

# force purge av modules (to avoid possible conflicts)
module --force purge

# load appropriate modules  (NIC5)
#module load releases/2020a
#module load releases/2021a
module load releases/2020b

#module load Python/3.8.2-GCCcore-9.3.0
#module load Python/3.9.5-GCCcore-10.3.0
#module load SciPy-bundle/2020.03-foss-2020a-Python-3.8.2

module load  Python/3.8.6-GCCcore-10.2.0


module load SciPy-bundle/2020.11-foss-2020b

#module load releases/2021a
#module load Python/3.9.5-GCCcore-10.3.0


ulimit -s unlimited 

# activate virtual environnement

#virtualenv --system-site-packages ~/my_venv_pip
#source ~/my_venv_pip/bin/activate


echo "Job starting at $(date)" 

### Commands to run your program start here #################################### 
# --show-leak-kinds=definite --log-file=valgrind-output python  test.py  
#PYTHONMALLOC=malloc valgrind --leak-check=full --log-file=valgrind-output python  test.py  


# Note that we do not pass a redis password. The number of cpus needs to match nnodes*cpus-per-task

python generate_data.py  

echo "Job end at $(date)"  
