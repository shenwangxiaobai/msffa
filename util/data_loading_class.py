import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '',type='train'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # self.img_transform = A.Compose([
        #     A.Cutout(num_holes=20, max_h_size=50, max_w_size=50, fill_value=0, p=0.6),
        #     A.GaussianBlur(blur_limit=11, always_apply=False, p=0.6),
        #     A.InvertImg(always_apply=False,p=0.6)
        # ])



        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((512, 512), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)


    def maskcalss(self,mask):
        mask = np.array(mask)
        mask[mask>0]=1
        return mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        assert len(mask_file) == 1, f'Either no test or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        mask3 = self.load(mask_file[0])
        img = self.load(img_file[0])


        assert img.size == mask.size, \
            'Image and test {name} should be the same size, but are {img.size} and {test.size}'

        imgs = self.preprocess(img, self.scale, is_mask=False)
        img50 = self.preprocess(img, 0.5, is_mask=False)
        img75 = self.preprocess(img, 0.75, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        mask_128 = self.preprocess(mask3, 0.25, is_mask=True)
        mask2 = self.maskcalss(mask)


        # if self.type == 'train':
        # img = self.img_transform(image=img)
        # img2 = img['image']
        # print(img)

        return {
            'image': torch.as_tensor(imgs.copy()).float().contiguous(),
            'image50': torch.as_tensor(img50.copy()).float().contiguous(),
            'image75': torch.as_tensor(img75.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'maskclass': torch.as_tensor(mask2.copy()).long().contiguous(),
            'maak128':torch.as_tensor(mask_128.copy()).long().contiguous(),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='')
