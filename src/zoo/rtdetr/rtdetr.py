"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register
# from .blip_vqa import BlipVQA


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['lighter', 'backbone', 'encoder', 'decoder', ]

    def __init__(self, lighter: nn.Module, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.lighter = lighter
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        # self.blipVQA = BlipVQA(device='cuda' if torch.cuda.is_available() else 'cpu')     # yes改
        
    def forward(self, x, targets=None):
        '''
            x.shape = torch.Size([4, 3, 480, 480])
            rtdetr backbone x[0].shape----------------- torch.Size([4, 512, 80, 80])   H、W每次不一样，可能和图片本身有关
            rtdetr backbone x[1].shape----------------- torch.Size([4, 1024, 40, 40])
            rtdetr backbone x[2].shape----------------- torch.Size([4, 2048, 20, 20])

            targets有时会有空样本，不知道为啥，似乎不影响检测
        '''

        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x_li = self.lighter(x, True)
        # descriptions = self.blipVQA.forward_npimg(x_li)
        # x_bout = self.backbone(x)
        x_libout = self.backbone(x_li)
        # con_x = x + x_li
        x_libout = self.encoder(x_libout)        
        x_libout = self.decoder(x_libout, targets)   # ---- error: target['desc'] is on cpu  ---- ok

        return x_libout  # yes改
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 

# num_pixels = x.size(1) * x.size(2) * x.size(3)
# rec_loss2 = torch.sum((x_li - x) ** 2.0) / num_pixels

# rec_loss1 = 0
# for i in range(3):
#     num_pixels2 = x_bout[i].size(1) * x_bout[i].size(2) * x_bout[i].size(3)
#     rec_loss1 += torch.sum((x_libout[i] - x_bout[i]) ** 2.0) / num_pixels2

# print("========rec_loss1=======", rec_loss1)
# print("========rec_loss2=======", rec_loss2)
# rec_loss = rec_loss1 + rec_loss2
# rec_loss = 5 * rec_loss
# print("========rec_loss=======", rec_loss)
# quit()

        # sim_loss *= 0.2
        # #-------------------------------------------
        # # 处理图像到B,T,256
        # batch_size, channels, height, width = x_li.shape
        # x_li = x_li.view(batch_size, channels, height * width)   # [B, C, H*W]
        # x_li = x_li.permute(0, 2, 1)  # [B, H*W, C]
        # img_projection = nn.Linear(channels, 256).to(x_li.device)
        # image_feats_proj = img_projection(x_li)
        # # 计算sim_loss
        # sim_loss = 0
        # textnorm = text_feats / text_feats.norm(dim=-1, keepdim=True)
        # imgnorm = image_feats_proj / image_feats_proj.norm(dim=-1, keepdim=True)
        # similarity = imgnorm[:,0,:] @ textnorm[:,0,:].t()
        # normalized_sim = (similarity + 1) / 2            
        # # 遍历相似度矩阵，选择正负样本对
        # for i in range(normalized_sim.size(0)):
        #     for j in range(normalized_sim.size(1)):          
        #         if i == j:  # ISP与原图应远离，负样本
        #             sim_loss += normalized_sim[i][j]     
        # #-------------------------------------------
