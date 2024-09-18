#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np
os.environ['ALBUMENTATIONS_SKIP_VERSION_CHECK'] = '1'
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from model import AttmNet
from PIL import Image
from dataset import get_train_val_test_loader_from_train, get_train_val_test_indices

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
            
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    model = AttmNet(in_chans=3,out_chans=1,depths=[2,2,2,2],feat_size=[48, 96, 192, 384])
    model = model.cuda()

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'BUSI':
        mask_ext = '_mask.png'
    elif dataset_name == 'PH2':
        mask_ext = '.png'
    elif dataset_name == 'BUS':
        mask_ext = '.png'
   
    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    # img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    data = '.../inputs/BUS/images'
    train_img_ids, val_img_ids, test_img_ids = get_train_val_test_indices(data)

    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')

    try:        
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)
        
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()
    specificity_avg_meter = AverageMeter()    
    precision_avg_meter = AverageMeter()
    f1_score_avg_meter = AverageMeter()

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou_, dice_, hd95_, recall_, specificity_, precision_, f1_score = indicators(output, target)   
            iou_avg_meter.update(iou_, input.size(0))
            dice_avg_meter.update(dice_, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))
            recall_avg_meter.update(recall_, input.size(0))
            specificity_avg_meter.update(specificity_, input.size(0))
            precision_avg_meter.update(precision_, input.size(0))
            f1_score_avg_meter.update(f1_score, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            for pred, img_id in zip(output, meta['img_id']):
                pred_np = pred[0].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))

    
    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('HD95: %.4f' % hd95_avg_meter.avg)
    print('recall: %.4f' % recall_avg_meter.avg)
    print('specificity: %.4f' % specificity_avg_meter.avg)
    print('precision: %.4f' % precision_avg_meter.avg)
    print('f1_score: %.4f' % f1_score_avg_meter.avg)




if __name__ == '__main__':
    main()
