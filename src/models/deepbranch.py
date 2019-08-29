import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from .finetune import finetune_vggresnet, finetune_vggsenet

from .attention import Attention
from catalyst.contrib.modules.pooling import GlobalConcatPool2d


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DeepBranchAttResnet(nn.Module):

    def __init__(self, pretrained=True, backbone=None, n_class=7, agg_mode='fc'):
        super(DeepBranchAttResnet, self).__init__()

        params = {
            "pretrained": pretrained,
            "n_class": n_class
        }
        base_model = finetune_vggresnet(**params)

        num_features = 256

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

        x = Variable(torch.rand(1, 3, 224, 224))

        for i in range(5):
            encoder = getattr(self, f"encoder{i}")
            x, conv_in_channels = self.get_conv_in_channels(x, encoder)
            conv = nn.Sequential(
                # GlobalConcatPool2d(),
                Flatten(),
                nn.BatchNorm2d(conv_in_channels * 2),
                nn.Dropout(0.2)
            )
            setattr(self, f"conv_block_{i}", conv)
            fc_in_channels = self.get_fc_in_channels(x, conv)
            fc = nn.Sequential(
                Flatten(),
                nn.Linear(fc_in_channels, num_features),
            )
            setattr(self, f"fc{i}", fc)
        # Last FC5
        self.fc5 = nn.Linear(
            in_features=base_model.fc.in_features,
            out_features=num_features
        )

        self.final_fc = nn.Linear(num_features, 7)

        self.model_name = 'vggresnet'
        self.att = Attention(256, 6)

    def forward(self, x):
        x0 = self.encoder0(x)
        x0_side = self.conv_block_0(x0)

        x1 = self.encoder1(x0)
        x1_side = self.conv_block_1(x1)

        x2 = self.encoder2(x1)
        x2_side = self.conv_block_2(x2)

        x3 = self.encoder3(x2)
        x3_side = self.conv_block_3(x3)

        x4 = self.encoder4(x3)
        x4_side = self.conv_block_4(x4)

        x5 = self.pool(x4)
        x5 = x5.view(x5.size(0), -1)

        logit0 = self.fc0(x0_side)
        logit1 = self.fc1(x1_side)
        logit2 = self.fc2(x2_side)
        logit3 = self.fc3(x3_side)
        logit4 = self.fc4(x4_side)
        logit5 = self.fc5(x5)

        logit = torch.cat([
            logit0, logit1,
            logit2, logit3,
            logit4, logit5
        ], -1)

        logit = logit.view(logit.size(0), -1, 256)

        logit = self.att(logit)
        logit = self.final_fc(logit)
        return logit

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


class DeepBranchResnet(DeepBranchAttResnet):

    def __init__(self, pretrained=True, n_class=7, agg_mode='fc'):
        super(DeepBranchResnet, self).__init__()

        params = {
            "pretrained": pretrained,
            "n_class": n_class
        }
        base_model = finetune_vggresnet(**params)

        kernel_size = 1
        out_channels = 1
        dropout = 0.2

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

        x = Variable(torch.rand(1, 3, 224, 224))

        for i in range(5):
            encoder = getattr(self, f"encoder{i}")
            x, conv_in_channels = self.get_conv_in_channels(x, encoder)
            conv = nn.Sequential(
                nn.Conv2d(in_channels=conv_in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout)
            )
            setattr(self, f"conv_block_{i}", conv)
            fc_in_channels = self.get_fc_in_channels(x, conv)
            fc = nn.Sequential(
                Flatten(),
                nn.Linear(fc_in_channels, n_class),
            )
            setattr(self, f"fc{i}", fc)
        # Last FC5
        self.fc5 = base_model.fc

        self.agg_mode = agg_mode
        if self.agg_mode == 'fc':
            self.final_fc = nn.Linear(n_class * 6, n_class)

        self.model_name = 'vggresnet'

    def forward(self, x):
        x0 = self.encoder0(x)
        x0_side = self.conv_block_0(x0)

        x1 = self.encoder1(x0)
        x1_side = self.conv_block_1(x1)

        x2 = self.encoder2(x1)
        x2_side = self.conv_block_2(x2)

        x3 = self.encoder3(x2)
        x3_side = self.conv_block_3(x3)

        x4 = self.encoder4(x3)
        x4_side = self.conv_block_4(x4)

        x5 = self.pool(x4)
        x5 = x5.view(x5.size(0), -1)

        logit0 = self.fc0(x0_side)
        logit1 = self.fc1(x1_side)
        logit2 = self.fc2(x2_side)
        logit3 = self.fc3(x3_side)
        logit4 = self.fc4(x4_side)
        logit5 = self.fc5(x5)

        if self.agg_mode == 'fc':
            logits = torch.cat([logit0, logit1, logit2, logit3, logit4, logit5], 1)
            return self.final_fc(logits)
        elif self.agg_mode == 'mean':
            return (5 * logit0 + 4 * logit1 + 3 * logit2 + 2 * logit3 + 1 * logit4 + logit5) / 16


"""
Deepbranch model from finetuned VGGResnet
"""
def deepbranch_vggresnet(params):
    return DeepBranchResnet(**params)
