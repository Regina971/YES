import torch
import torchvision.transforms as T
import numpy as np
from numpy import random as R
import cv2
import random
import scipy.stats as stats
from src.core import register
import uuid

__all__ = ['IlluminationAdjusting']
# 保存图像
def save_tensor_image(tensor, filename):
    # 将张量保存为图像
    tensor = T.ToPILImage()(tensor)
    tensor.save(filename)
    print("Saving" + filename)

def apply_ccm(image, ccm):
    '''
    The function of apply CCM matrix
    '''
    shape = image.shape
    image = image.view(-1, 3)
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    return image.view(shape)

class ImprovedWB(torch.nn.Module):
    def __init__(self):
        super(ImprovedWB, self).__init__()
        self.channels = 3
        self.log_wb_range = 0.5
        self.white_balance_params = torch.nn.Parameter(torch.ones(self.channels, dtype=torch.float32))  # 初始化白平衡参数

    def filter_param_regressor(self):
        mask = torch.tensor([[0, 1, 1]], dtype=torch.float32, device=self.white_balance_params.device)  # shape (1,3)
        features = self.white_balance_params * mask
        tanh_features = torch.tanh(features * self.log_wb_range)
        color_scaling = torch.exp(tanh_features)

        # normalize by luminance
        normalization_factor = 1.0 / (
            1e-5 + 0.27 * color_scaling[:,0] + 0.67 * color_scaling[:,1] + 0.06 * color_scaling[:,2]
        )
        color_scaling1 = color_scaling * normalization_factor

        return color_scaling1

    def apply_white_balance(self, img):
        '''
        img shape: torch.Size([800, 800, 3]) (H, W, C)
        param shape: torch.Size([1, 3]) (1, C)
        '''

        param = self.filter_param_regressor()
        param1 = param.unsqueeze(0)

        img_adjusted = img * param1

        return img_adjusted

@register
class IlluminationAdjusting(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 定义可学习参数
        self.lightgamma = torch.nn.Parameter(torch.tensor(0.8))
        self.white_balance_filter = ImprovedWB()

    def forward(self, imgs, safe_invert=False, targets=None):
        # imgs: (B, C, H, W)
        '''
        Simple pipeline: invert tone, WB, gamma
        '''
        save_path = "src/zoo/rtdetr/try_ispout/"
        batch_size = imgs.shape[0]
        img_high = torch.empty_like(imgs)
        
        device = imgs.device
        epsilon = torch.tensor(1e-8, device=device)

        for b in range(batch_size):

            img = imgs[b].permute(1, 2, 0)  # (C, H, W) -- (H, W, C)
            if targets is not None:
                image_id = targets[b]["image_id"]
            else:
                image_id = uuid.uuid1()

            '''inverse tone mapping'''
            img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img) / 3.0)
            # save_tensor_image(img1.permute(2, 0, 1), save_path + str(image_id) + "_itm.jpg")
            # img1 = img

            '''white balance'''
            img2 = self.white_balance_filter.apply_white_balance(img1).to(img1.device)
            # save_tensor_image(img2.permute(2, 0, 1), save_path + str(image_id) + "_wb.jpg")

            '''gamma correction'''
            img3 = torch.max(img2, epsilon) ** (1 / self.lightgamma)
            # save_tensor_image(img3.permute(2, 0, 1), save_path + str(image_id) + "_gc.jpg")

            img3 = torch.clamp(img3, 0, 0.8) 

            img_high[b] = img3.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)

        return img_high
        