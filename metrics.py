import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (output_.sum() + target_.sum() + smooth)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    #hd_ = hd(output_, target_)
    if np.sum(output_) == 0 or np.sum(target_) == 0:
        print("Warning: One of the arrays is empty. Skipping HD calculation.")
        hd_ = 0  
    else:
        hd_ = hd(output_, target_)
    if np.sum(output_) == 0 or np.sum(target_) == 0:
        print("Warning: One of the arrays is empty. Skipping HD calculation.")
        hd95_ = 0  
    else:
        hd95_ = hd95(output_, target_)

    #hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    # Calculate F1 score
    f1_score = 2 * (precision_ * recall_) / (precision_ + recall_ + 1e-5)

    return iou_, dice_, hd95_, recall_, specificity_, precision_, f1_score

