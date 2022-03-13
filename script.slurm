#!/bin/bash

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --job-name="in_pre"
#SBATCH --account=punim1623
#SBATCH --time=12:00:00


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=100G

#SBATCH --mail-user=haitianh@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load foss/2019b
module load web_proxy
module add cuda/10.1.243
module add gcc/8.3.0
# module load pytorch/1.5.1-python-3.7.4
# module load numpy/1.18.0-python-3.7.4

module load anaconda3/2021.05
source activate pixpro

./tools/pixpro_imagenet.sh