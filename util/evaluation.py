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
    print(iou)
    miou = np.nanmean(iou)

    return miou
if __name__=='__main__':
    path1 = r'./data/file-0007_0000.nii.gz'
    path2 = r'./data/gt/file-0007_0000.nii.gz'
    imgPredict = np.asarray(nib.load(path1).get_fdata())
    imgLabel = np.asarray(nib.load(path2).get_fdata())



    # 计算HD95，dice
    current_hd95,current_dc = calculate_metric_percase(imgPredict, imgLabel)
    print('hd95', current_hd95, 'dc', current_dc)

    # 计算MIOU，iou
    hist = generate_matrix(16, imgPredict, imgLabel)
    miou_res = miou(hist)
    print(miou_res)