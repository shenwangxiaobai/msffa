import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Up(nn.Module):
    """Upscaling and concat"""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x


class NestedUNet(nn.Module):
    def __init__(self, num_classes=16, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = Up()

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # import pdb
        # pdb.set_trace()
        # input = input.repeat(1,3,1,1)
        # print(input.shape)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(self.up(x1_0, x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(self.up(x2_0, x1_0))
        x0_2 = self.conv0_2(self.up(x1_1, torch.cat([x0_0, x0_1], 1)))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(self.up(x3_0, x2_0))
        x1_2 = self.conv1_2(self.up(x2_1, torch.cat([x1_0, x1_1], 1)))
        x0_3 = self.conv0_3(self.up(x1_2, torch.cat([x0_0, x0_1, x0_2], 1)))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.up(x4_0, x3_0))
        x2_2 = self.conv2_2(self.up(x3_1, torch.cat([x2_0, x2_1], 1)))
        x1_3 = self.conv1_3(self.up(x2_2, torch.cat([x1_0, x1_1, x1_2], 1)))
        x0_4 = self.conv0_4(self.up(x1_3, torch.cat([x0_0, x0_1, x0_2, x0_3], 1)))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


dir_img = Path('/home/zhoujing/biyeba/data/data3/imgs')
dir_mask = Path('/home/zhoujing/biyeba/data/data3/mask')

dir_img_v = Path('/home/zhoujing/biyeba/data/data3/test/imgs')
dir_mask_v = Path('/home/zhoujing/biyeba/data/data3/test/mask')

dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 200,
              batch_size: int = 4,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, 1.0)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, 1.0)

    n_val = int(len(dataset) * 0)
    n_train = len(dataset) - n_val
    train_set, _ = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))


    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(16)


    net.train()
    net.to(device=device)

    base_lr = 3e-4

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = 200
    max_iterations = 200 * len(dataset)
    iter_num = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
       
        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)


            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)

                import pdb

                # pdb.set_trace()

                loss_ce = ce_loss(masks_pred, true_masks[:].long())
                loss = loss_ce

                print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

                iter_num = iter_num + 1

        if epoch%20 == 0 or epoch > epochs-5:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            # logging.info(f'Checkpoint {epoch + 1} saved!')



if __name__ == '__main__':
    #tensorboard --logdir logs_model
    net = NestedUNet()
    writer = SummaryWriter("logs_model")
    input = torch.ones(( 32,3, 256, 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_net(net=net,device=device)

    # writer.add_graph(net, input)
    # writer.close()
