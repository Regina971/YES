#!/bin/bash
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_4090
#SBATCH -J ispout
#SBATCH -o output/ispout-%J.log

module load cuda/11.8
module load cudnn/8.5.0_cuda11.x
module load anaconda/2022.10
source activate rtdetr

echo Start slurm job at `date`

export CUDA_VISIBLE_DEVICES=0
# python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -t /HOME/scw6c14/run/RT-DETR/rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth
python tools/train.py -c configs/rtdetr/rtdetr_exdark.yml -t exdark_final/newWB_B_sinISP_CPW/checkpoint0027.pth --test-only

echo End slurm job at $(date)