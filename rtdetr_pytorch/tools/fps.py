import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from src.core import YAMLConfig
from pathlib import Path
import time

cla = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']
color = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0)]

class ImageReader:
    def __init__(self, resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None   #保存最近一次读取的图片的pil对象

    def __call__(self, image_path, *args, **kwargs):
        """
        读取图片
        """
        self.pil_img = Image.open(image_path).convert('RGB').resize((self.resize, self.resize))
        return self.transform(self.pil_img).unsqueeze(0)




class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu') 
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)



def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtdetr/rtdetr_exdark.yml", help="配置文件路径")
    parser.add_argument("--ckpt", default="output/rtdetr_r50vd_6x_coco/checkpoint0021.pth", help="权重文件路径")
    # parser.add_argument("--image", default="datasets/ExDark_coco/origin_test/2015_07021.jpg", help="待推理图片路径")
    parser.add_argument("--image_dir", default="datasets/ExDark_coco/origin_test", help="待推理文件夹路径")
    parser.add_argument("--output_dir", default="output/rtdetr_r50vd_6x_coco/infer/origin", help="输出文件保存路径")
    parser.add_argument("--device", default="cuda")

    return parser

def main(args):
    ''' 批量推理多张图片 '''
    image_dir = Path(args.image_dir)
    device = torch.device(args.device)
    reader = ImageReader(resize=640)
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)
    model.eval()  

    batch_size = 8  
    total_time = 0  
    num_images = 0  
    batch_imgs = []  
    for img_path in image_dir.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img = reader(img_path).to(device)
            if img.dim() == 4: 
                img = img.squeeze(0)  # 从 [1, 3, H, W] -> [3, H, W]
            batch_imgs.append(img)
            
            # 当批量大小达到设定值时，执行推理
            if len(batch_imgs) == batch_size:
                batch = torch.stack(batch_imgs)  
                batch = batch.to(device) 
                size = torch.tensor([[batch.shape[2], batch.shape[3]]] * batch_size).to(device)
                
                # 推理并计时
                start = time.time()
                output = model(batch, size)
                elapsed_time = time.time() - start
                total_time += elapsed_time
                num_images += batch_size
                print(f"批量推理耗时：{elapsed_time:.4f}s，处理 {batch_size} 张图片")
                
                batch_imgs = [] 
        else:
            print(f"跳过不支持的图片格式: {img_path}")

    # 处理最后不足一个批次的图片
    if batch_imgs:
        batch = torch.stack(batch_imgs)
        batch = batch.to(device)
        size = torch.tensor([[batch.shape[2], batch.shape[3]]] * len(batch_imgs)).to(device)
        
        start = time.time()
        output = model(batch, size)
        elapsed_time = time.time() - start
        total_time += elapsed_time
        num_images += len(batch_imgs)
        print(f"最后一批推理耗时：{elapsed_time:.4f}s，处理 {len(batch_imgs)} 张图片")

    # 计算 FPS
    if num_images > 0:
        fps = num_images / total_time
        print(f"总图片数：{num_images}，总耗时：{total_time:.4f}s，平均 FPS：{fps:.2f}")
    else:
        print("未处理任何图片。")


if __name__ == "__main__":
    main(get_argparser().parse_args())
    

