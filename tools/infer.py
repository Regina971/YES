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
        self.pil_img = None 

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
    parser.add_argument("--config", default="configs/rtdetr/rtdetr_r50vd_6x_coco.yml", help="配置文件路径")
    parser.add_argument("--ckpt", default="output/rtdetr_r50vd_6x_coco/checkpoint0021.pth", help="权重文件路径")
    # parser.add_argument("--image", default="datasets/ExDark_coco/origin_test/2015_07021.jpg", help="待推理图片路径")
    parser.add_argument("--image_dir", default="datasets/ExDark_coco/try", help="待推理文件夹路径")
    parser.add_argument("--output_dir", default="infer/561826644", help="输出文件保存路径")
    parser.add_argument("--device", default="cuda")

    return parser

def main(args):
    ''' 推理多张图片 '''
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    reader = ImageReader(resize=640)
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)

    for img_path in image_dir.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img = reader(img_path).to(device)
            size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)
            start = time.time()
            output = model(img, size)
            print(f"推理耗时：{time.time() - start:.4f}s")
            labels, boxes, scores = output
            im = reader.pil_img
            draw = ImageDraw.Draw(im)
            thrh = 0.6

            for i in range(img.shape[0]):
                scr = scores[i]
                lab = labels[i][scr > thrh]
                box = boxes[i][scr > thrh]

                for idx, b in enumerate(box):
                    text_size = draw.textsize(cla[lab[idx]])
                    text_box = [b[0], b[1], b[0], b[1]]
                    draw.rectangle(text_box, fill=color[lab[idx]])

                    draw.rectangle(list(b), outline=color[lab[idx]], width=3)
                    draw.text((b[0], b[1]), text=cla[lab[idx]], fill='white', )

            save_path = output_dir / img_path.name
            im.save(save_path)
            print(f"检测结果已保存至:{save_path}")
        else:
            print(f"跳过不支持的图片格式: {img_path}")

if __name__ == "__main__":
    main(get_argparser().parse_args())
    

