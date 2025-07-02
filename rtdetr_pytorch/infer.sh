#!/bin/bash
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu_4090
#SBATCH -J try
#SBATCH -o infer/try-%J.log

module load cuda/11.8
module load cudnn/8.5.0_cuda11.x
module load anaconda/2022.10
source activate rtdetr

echo Start slurm job at `date`

# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0
# python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -t /HOME/scw6c14/run/RT-DETR/rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth
python tools/infer.py --config configs/rtdetr/rtdetr_exdark.yml --ckpt exdark_final/newWB_B_sinISP_CPW/checkpoint0027.pth --image_dir /HOME/scw6c14/run/datasets/ExDark_coco/try --output_dir infer/try

echo End slurm job at $(date)