#!/bin/bash
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_4090
#SBATCH -J clamp0.9
#SBATCH -o impro_exdark/clamp0.9-%J.log

module load cuda/11.8
module load cudnn/8.5.0_cuda11.x
module load miniforge/24.1.2
source activate rtdetr

echo Start slurm job at `date`

export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_exdark.yml -t rtdetr_r50vd_6x_coco_from_paddle.pth
# exdark_final/finalgamma08_B_newsimISP_CPW/checkpoint0036.pth
# /HOME/scw6c14/run/RT-DETR/rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth

echo End slurm job at $(date)