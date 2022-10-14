import random
import argparse
import logging
import sys

import os
from pathlib import Path
from torch.autograd import Variable
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from util.data_loading_class import BasicDataset, CarvanaDataset
from util.dice_score import DiceLoss
from evaluate_mult2 import evaluate
from SemanticSegmentation import MSFFA as ModelT




dir_img = Path('/home/zhoujing/biyeba/data/data3/imgs/')
dir_mask = Path('/home/zhoujing/biyeba/data/data3/mask/')

dir_img2 = Path('/home/zhoujing/biyeba/data/data3/test/imgs/')
dir_mask2 = Path('/home/zhoujing/biyeba/data/data3/test/mask/')

dir_img3 = Path('/home/zhoujing/biyeba/data/data3/val/imgs/')
dir_mask3 = Path('/home/zhoujing/biyeba/data/data3/val/mask/')
#

dir_checkpoint = Path('./checkpoints_UnetClass1/')


def train_net(net,
              device,
              epochs: int = 200,
              batch_size: int = 3,
              learning_rate: float = 0.001,
              val_percent: float = 0,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    try:
        vasldataset = CarvanaDataset(dir_img2, dir_mask2, img_scale)
    except (AssertionError, RuntimeError):
        vasldataset = BasicDataset(dir_img2, dir_mask2, img_scale)

    try:
        testdataset = CarvanaDataset(dir_img3, dir_mask3, img_scale)
    except (AssertionError, RuntimeError):
        testdataset = BasicDataset(dir_img3, dir_mask3, img_scale)

    alpha = 8/255


    train_set = dataset
    val_set = vasldataset

    n_train = len(dataset)
    n_val = len(vasldataset)



    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)
    test_loader = DataLoader(testdataset, shuffle=False, drop_last=False, **loader_args)

    loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net++', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        alpha:           {str(alpha)}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    ce_loss = nn.CrossEntropyLoss()
    ce_loss1 = DiceLoss(1)
    dice_loss = DiceLoss(7)

    # 5. Begin training
    epochs_loss = []
    ym = 0

    t = 2

    lmlr = learning_rate
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        Traintimes = []

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            start = time.perf_counter()

            for batch in train_loader:

                # 中间写上代码块

                images = batch['image']
                images50 = batch['image50']
                images75 = batch['image75']
                # images2 = batch['image2']
                true_masks = batch['mask']
                class_mask = batch['maskclass']

                # images = images.repeat(1, 3, 1, 1)
                # images50 = images50.repeat(1, 3, 1, 1)
                # images75 = images75.repeat(1, 3, 1, 1)

                assert images.shape[1] == net.input_channels, \
                    f'Network has been defined with {net.input_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                images50 = images50.to(device=device, dtype=torch.float32)
                images75 = images75.to(device=device, dtype=torch.float32)
                # images2 = images2.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # mask128 = mask128.to(device=device, dtype=torch.long)
                class_mask = class_mask.to(device=device, dtype=torch.long)
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss_ce = ce_loss(masks_pred[:], true_masks[:].long())
                    # loss_as = ce_loss1(mask_class, class_mask, softmax=True)
                    loss_dice = dice_loss(masks_pred, true_masks, softmax=True)
                    # loss128 = dice_loss(mask_128, mask128, softmax=True)
                    # loss = 0.5 * loss_ce + 0.5 * loss_dice
                    loss = 0.5 * loss_ce + 0.5 * loss_dice
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            end = time.perf_counter()
            Traintimes.append([end - start])
            logging.info({'Train time': end - start})
            logging.info({'epoch loss': epoch_loss / 175,'lr':optimizer.state_dict()['param_groups'][0]['lr']})
            epochs_loss.append([epoch_loss / 175])

            test_score = evaluate(net, test_loader, device)
            # scheduler.step(test_score)

            # Evaluation round
            division_step = (n_train // (1 * n_train))
            if 0 == 0:
                if 0 == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        # tag = tag.replace('/', '.')
                        # histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        # histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        pass

                    val_score = evaluate(net, val_loader, device)
                    # scheduler.step(val_score)

                    logging.info('Validation Dice score: {}'.format(val_score))
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'test dice':test_score,
                        'validation Dice': val_score,
                        'images': wandb.Image(images[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu()),
                            'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

        if save_checkpoint and (epoch + 1) % 10 == 0 :
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
            # logging.info(f'per time', np.mean(np.array(Traintimes)))
            pd.DataFrame(np.array(epochs_loss).flatten()).to_csv(str(dir_checkpoint) + 'Lightmodels.csv', header=None,
                                                                 index=0)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=6, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load Lightmodels from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = ModelT(num_classes=7)

    for name, value in torch.load(
            '/home/aplysia/helloWorld/pytorch-networks/checkpoints_UnetClass1/checkpoint_epoch100.pth').items():
        # print(model.state_dict()[name].shape)
        if net.state_dict()[name].shape == value.shape:
            net.state_dict()[name] = value
        else:
            net.state_dict()[name] = torch.ones(value.shape)

    logging.info(f'Network:\n'
                 f'\t{net.input_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.deep_supervision else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
