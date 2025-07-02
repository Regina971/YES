"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
import copy
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
from src.zoo.rtdetr.high_upgrade import IlluminationAdjusting
from PIL import Image
import torchvision.transforms.functional as TF

def get_list_dimensions(lst):
    if not isinstance(lst, list):
        return []
    
    dimensions = []
    while isinstance(lst, list):
        dimensions.append(len(lst))
        lst = lst[0] if len(lst) > 0 else None
    
    return dimensions

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        ### 一个batch的训练
        samples = samples.to(device)

        # ### 若有desc字段，则提取出来（train--true  val--false）  ##yes改
        # if ('desc' in target.keys() for target in targets):  
        #     descriptions = [target.pop('desc', [""]) for target in targets]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:    ### scaler == None
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            
            # # print("samples type:", type(samples))  ## Tensor
            # # print("targets type:", type(targets))  ## List
            # print("========samples shape=========", samples.shape)   ## B,C,H,W
            # quit()
            # lighter = IlluminationAdjusting().to(device)            
            # batched_inputs_light = copy.deepcopy(samples)
            # para_gt = torch.empty(size=(batch_size, 4)).to(device)
            


            # print("------------------")
            # print("para_gt[1](darkness, gamma value, WB red, WB blue):", para_gt[1])
            # print("trying to save imgs...")
            # batched_inputs_dark[1] = torch.clamp(batched_inputs_dark[1], 0, 1)   
            # # 转换输出图像张量为 PIL 图像
            # samples_pil = TF.to_pil_image(samples[1])
            # dark_pil = TF.to_pil_image(batched_inputs_dark[1])
            # # 保存输出图像为 JPG 文件
            # samples_pil.save("/HOME/scw6c14/run/lol-det/lilo-detr/rtdetr_pytorch/src/solver/samples_pil.jpg")
            # dark_pil.save("/HOME/scw6c14/run/lol-det/lilo-detr/rtdetr_pytorch/src/solver/dark_pil.jpg")
            # print("------------------")

            # quit()
            # outputs = model(samples, targets, descriptions)
            outputs = model(samples, targets)
            # print("----------det_engine train one epoch outputs---------", outputs)   ## 正常

            loss_dict = criterion(outputs, targets)
            
            # loss_dict['sim_loss'] = sim_loss
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )
    # print("-------------det_engine data_loader-----------------", data_loader)
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # ### 若有desc字段，则提取出来（train--true  val--false）
        # if ('desc' in target.keys() for target in targets):  
        #     descriptions = [target.pop('desc', [""]) for target in targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)      
        # print("----------det_engine evaluate outputs---------", outputs)  ## 只有一个单词 ---ok
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)


        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        # coco_evaluator.getapbyclass()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


# for name, parameter in model.named_parameters():
#     if 'decoder.fusion_model' in name or 'encoder.fpn_opt_text' in name or 'encoder.pan_opt_text' in name:
#         # print("Set" + name + "not to grad")
#         parameter.requires_grad = False



