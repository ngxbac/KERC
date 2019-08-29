import torch
from torch.autograd import Variable
import torch.nn as nn
# from .finetune import finetune_vggresnet
from models.finetune import finetune_vggresnet


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class RDBBottneck(nn.Module):
    def __init__(self, block):
        super(RDBBottneck, self).__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1

        self.conv2 = block.conv2
        self.bn2 = block.bn2

        self.conv3 = block.conv3
        self.bn3 = block.bn3

        if hasattr(block, 'downsample'):
            self.downsample = block.downsample
        else:
            self.downsample = None

        self.relu = block.relu
        channel_0 = self.conv1.in_channels
        channel_1 = self.conv1.out_channels
        channel_2 = self.conv1.out_channels
        channel_3 = self.conv3.out_channels
        concat_channels = channel_1 + channel_2

        self.conv = nn.Conv2d(in_channels=concat_channels, out_channels=channel_3, kernel_size=1)

    def forward(self, x):
        # import pdb
        # pdb.set_trace()

        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)

        out = torch.cat([out1, out2], 1)
        out = self.conv(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out3 += residual
        out3 += out
        out3 = self.relu(out3)

        return out3


class RDBResnet(nn.Module):

    def __init__(self, pretrained=True, backbone=None, n_class=7):
        super(RDBResnet, self).__init__()

        params = {
            "pretrained": pretrained,
            "n_class": n_class,
            "backbone": backbone
        }
        base_model = finetune_vggresnet(params)

        self.encoder0 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4
        self.pool = base_model.avgpool

        rdb_blocks = [RDBBottneck(block) for block in self.encoder1]
        self.rdb1 = nn.Sequential(*rdb_blocks)

        rdb_blocks = [RDBBottneck(block) for block in self.encoder2]
        self.rdb2 = nn.Sequential(*rdb_blocks)

        rdb_blocks = [RDBBottneck(block) for block in self.encoder3]
        self.rdb3 = nn.Sequential(*rdb_blocks)

        rdb_blocks = [RDBBottneck(block) for block in self.encoder4]
        self.rdb4 = nn.Sequential(*rdb_blocks)

        # Last FC5
        self.fc = base_model.fc

    def forward(self, x):
        x = self.encoder0(x)
        x = self.rdb1(x)
        x = self.rdb2(x)
        x = self.rdb3(x)
        x = self.rdb4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_fc_in_channels(self, x, block):
        with torch.no_grad():
            x = block(x)
            x = x.view(x.size(0), -1)
            return x.size(1)

    def get_conv_in_channels(self, x, block):
        with torch.no_grad():
            x = block(x)
            return x, x.size(1)

    def freeze_base(self):
        for module in [self.encoder0, self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_base(self):
        for module in [self.encoder0, self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
            for param in module.parameters():
                param.requires_grad = True


def rdb_resnet(params):
    return RDBResnet(**params)


if __name__ == '__main__':
    model = RDBResnet(pretrained=True, backbone=None, n_class=7)
    x = torch.rand((1, 3, 224, 224))
    out = model(x)