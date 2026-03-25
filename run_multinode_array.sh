#!/bin/bash
#SBATCH --job-name=tetra
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a.out

module purge
module load python/3.11

# activate your venv
source ~/venvs/reza/bin/activate

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python thermal_rho5.py
