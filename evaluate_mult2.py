from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from util.dice_score import multiclass_dice_coeff, dice_coeff
import nibabel as nib
import numpy as np
from medpy import metric

def calculate_metric_percase(pred, gt):
    hd95 = metric.binary.hd95(pred, gt)
    dc = metric.binary.dc(pred, gt)
    return hd95, dc

def generate_matrix(num_class, gt_image, pre_image):
    # 正确的gt_mask
    mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
    label = num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)  # (n, n)
    return confusion_matrix




def miou(hist):
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # print(iou)
    miou = np.nanmean(iou)

    return iou,miou

def evaluate(net,  dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # miou = 0

    DICE = []
    MIOU = []
    MIOULIST = []
    HD95 = []
    # iterate over the validation set

    for batch in dataloader:
        #tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # images50 = batch['image50']
        # images75 = batch['image75']
        # mask_true =  batch['maskclass']

        image = image.repeat(1, 3, 1, 1)
        # images50 = images50.repeat(1, 3, 1, 1)
        # images75 = images75.repeat(1, 3, 1, 1)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        # images50 = images50.to(device=device, dtype=torch.float32)
        # images75 = images75.to(device=device, dtype=torch.float32)
        # image = image.repeat(1, 3, 1, 1)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true2 = mask_true
        mask_true = F.one_hot(mask_true, net.num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.num_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.num_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 0:, ...], mask_true[:, 0:, ...], reduce_batch_first=False)

            mask_pred = mask_pred.cpu().numpy()
            mask_pred2 = np.argmax(mask_pred,axis=1)

            mask_true2 = mask_true2.cpu().numpy()


            # print(mask_pred2)
            # mask_true = np.array(mask_true).astype(np.float)
            ####################################################################################
            # current_hd95, current_dc = calculate_metric_percase(mask_pred, mask_true)

            # print(np.unique(mask_true))
            # print('hd95', current_hd95, 'dc', current_dc)
            # 计算MIOU，iou
            # mask_pred = np.argmax(mask_pred)
            # print(mask_true2.shape)
            # print(mask_pred.shape)
            hist = generate_matrix(16, mask_true2 , mask_pred2)
            iou,miou_res = miou(hist)
            # print('miou_res',miou_res)

            MIOULIST.append(iou)
            MIOU.append([miou_res])
            # print(MIOULIST)
    miousl = np.mean(np.array(MIOULIST),axis=0)
    miouse = np.mean(np.array(MIOU))

    print('miousl',miousl)
    print('miouse',miouse)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches



if __name__ == '__main__':

    def iou_mean(pred, target, n_classes=15):
        # n_classes ：the number of classes in your dataset,not including background
        # for mask and ground-truth label, not probability map
        ious = []
        iousSum = 0
        # pred = torch.from_numpy(pred)
        pred = pred.view(-1)
        # target = np.array(target)
        # target = torch.from_numpy(target)
        target = target.view(-1)

        # Ignore IoU for background class ("0")
        for cls in range(0, n_classes + 1):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
            union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(max(union, 1)))
                iousSum += float(intersection) / float(max(union, 1))
        return iousSum / n_classes

    dir_img2 = Path('/home/zhoujing/biyeba/Mine_Fed/data/huaxi350_resize/test/imgs/')
    dir_mask2 = Path('/home/zhoujing/biyeba/Mine_Fed/data/huaxi350_resize/test/mask/')

    pthdir = './checkpoints_aug_unetsedeep/checkpoint_epoch200.pth'
    pthdir = './checkpoints_unetunet/checkpoint_epoch200-1.pth'
    pthdir = './checkpoints_unetunet/checkpoint_epoch200.pth'


    pthdir = './pth_for_a_in_aaa/checkpoint_epoch200.pth'
    pthdir = './pth_for_a_in_bbb/checkpoint_epoch200.pth'
    pthdir = './pth_for_a_in_bbb1/checkpoint_epoch200.pth'
    pthdir = './pth_for_a_in_bbb2/checkpoint_epoch200.pth'

    # from SemanticSegmentation import DeepLabv3Plus34Res as Model
    from SemanticSegmentation import a001_2 as Model


    from util.data_loading_class import BasicDataset, CarvanaDataset
    from torch.utils.data import DataLoader, random_split

    try:
        vasldataset = CarvanaDataset(dir_img2, dir_mask2, 1)
    except (AssertionError, RuntimeError):
        vasldataset = BasicDataset(dir_img2, dir_mask2, 1)
    val_set = vasldataset

    loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)



    net = Model(num_classes=7)
    net.eval()
    net.load_state_dict(torch.load(pthdir))

    dic = evaluate(net,val_loader,device='cpu')
    print('dic',dic)