#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --job-name=MyJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err


module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate pytorch-CycleGAN-and-pix2pix

python metrics.py ./results/cityscapes_pix2pix_VGG/test_latest/images/ ./results/cityscapes_pix2pix_VGG/test_latest/
python metrics.py ./results/cityscapes_pix2pix_WGAN/test_latest/images/ ./results/cityscapes_pix2pix_WGAN/test_latest/
python metrics.py ./results/cityscapes_pix2pix_ResNet/test_latest/images/ ./results/cityscapes_pix2pix_ResNet/test_latest/