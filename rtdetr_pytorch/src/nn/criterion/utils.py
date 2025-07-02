import torch 
import torchvision



def format_target(targets):
    '''
    Args:
        targets (List[Dict]),
    Return: 
        tensor (Tensor), [im_id, label, bbox,]
    '''

    outputs = []
    for i, tgt in enumerate(targets):
        boxes =  torchvision.ops.box_convert(tgt['boxes'], in_fmt='xyxy', out_fmt='cxcywh') 
        labels = tgt['labels'].reshape(-1, 1)
        im_ids = torch.ones_like(labels) * i
        outputs.append(torch.cat([im_ids, labels, boxes], dim=1))

    return torch.cat(outputs, dim=0)


    ###----------------yes modified------------------gpt说不影响outputs的内容
    # outputs = []
    # descs = []
    
    # for i, tgt in enumerate(targets):
    #     num_boxes = tgt['boxes'].shape[0]
    #     im_ids = torch.ones(num_boxes, dtype=torch.long) * i
    #     labels = tgt['labels'].reshape(-1, 1)
    #     boxes = torchvision.ops.box_convert(tgt['boxes'], in_fmt='xyxy', out_fmt='cxcywh')
        
    #     # 添加每个目标的描述到列表中
    #     desc_list = [tgt['desc']] * num_boxes
    #     descs.extend(desc_list)
        
    #     # 将图像ID、标签和边界框拼接成一个张量
    #     output = torch.cat([im_ids.unsqueeze(1), labels, boxes], dim=1)
    #     outputs.append(output)
        
    # # 将所有目标拼接成一个大的张量，并返回
    # return torch.cat(outputs, dim=0), descs
    ###----------------yes modified------------------

