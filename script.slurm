#!/bin/bash

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --job-name="in_pre"
#SBATCH --account=punim1623
#SBATCH --time=0:10:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=32G

#SBATCH --mail-user=haitianh@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module purge
module load cuda/11.1.1

eval "$(conda shell.bash hook)"
conda activate newpixpro

./tools/pixpro_mvtec.sh