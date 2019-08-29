import torch
from torch.autograd import Variable
import torch.nn as nn
from models.finetune import finetune_vggresnet


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DSResnet(nn.Module):

    def __init__(self, pretrained=True, n_class=7):
        super(DSResnet, self).__init__()

        params = {
            "pretrained": pretrained,
            "n_class": n_class
        }
        base_model = finetune_vggresnet(**params)

        kernel_size = 3
        out_channels = 64
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
        self.model_name = 'dsvggresnet'

        self.fc_concat = nn.Linear(in_features=n_class * 6, out_features=n_class)

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

        x = torch.cat([
            logit0, logit1, logit2, logit3, logit4, logit5
        ], 1)

        x = self.fc_concat(x)

        return logit0, logit1, logit2, logit3, logit4, logit5, x


    def get_fc_in_channels(self, x, block):
        with torch.no_grad():
            x = block(x)
            x = x.view(x.size(0), -1)
            return x.size(1)

    def get_conv_in_channels(self, x, block):
        with torch.no_grad():
            x = block(x)
            return x, x.size(1)


class SSEResnet(nn.Module):

    def __init__(self, pretrained=True, n_class=7):
        super(SSEResnet, self).__init__()

        params = {
            "pretrained": pretrained,
            "n_class": n_class
        }
        base_model = finetune_vggresnet(**params)

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

        self.ss_block1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1),
            Flatten(),
            nn.Linear(in_features=2048, out_features=n_class)
        )

        self.is_block1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1),
            Flatten(),

            nn.Linear(in_features=2048, out_features=n_class)
        )

        self.is_block2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7, stride=1),
            Flatten(),

            nn.Linear(in_features=2048, out_features=n_class)
        )

        self.fc5 = nn.Linear(in_features=2048, out_features=n_class)
        self.fc_concat = nn.Linear(in_features=n_class * 4, out_features=n_class)

    def forward(self, x):
        x0 = self.encoder0(x)

        x1 = self.encoder1(x0)
        x1_side = self.ss_block1(x1)
        # import pdb
        # pdb.set_trace()

        x2 = self.encoder2(x1)
        x2_side = self.is_block1(x2)

        x3 = self.encoder3(x2)
        x3_side = self.is_block2(x3)

        x4 = self.encoder4(x3)
        x5 = self.pool(x4)
        x5 = x5.view(x5.size(0), -1)
        logit5 = self.fc5(x5)

        x_concat = torch.cat([
            x1_side, x2_side, x3_side, logit5
        ], 1)

        x_concat = self.fc_concat(x_concat)

        return x1_side, x2_side, x3_side, x5, x_concat


class SpatialAttn(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # e.g. 32,192
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0),1,h,w)
        return z


# Parameter-Free Spatial Attention Network for Person Re-Identification
class DSResnetSA(nn.Module):

    def __init__(self, pretrained=True, n_class=7):
        super(DSResnetSA, self).__init__()

        params = {
            "pretrained": pretrained,
            "n_class": n_class
        }
        base_model = finetune_vggresnet(**params)

        kernel_size = 3
        out_channels = 64
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

        self.SA0 = SpatialAttn()
        self.SA1 = SpatialAttn()
        self.SA1 = SpatialAttn()
        self.SA2 = SpatialAttn()
        self.SA3 = SpatialAttn()
        self.SA4 = SpatialAttn()

        for i in range(5):
            encoder = getattr(self, f"encoder{i}")
            x, conv_in_channels = self.get_conv_in_channels(x, encoder)
            conv = nn.Sequential(
                nn.Conv2d(in_channels=conv_in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout),
                nn.AdaptiveMaxPool2d(1),
            )
            setattr(self, f"conv_block_{i}", conv)
            # fc_in_channels = self.get_fc_in_channels(x, conv)
            fc = nn.Sequential(
                Flatten(),
                nn.Linear(out_channels, n_class),
            )
            setattr(self, f"fc{i}", fc)
        # Last FC5
        self.fc5 = base_model.fc
        self.model_name = 'dsvggresnet'

    def forward(self, x):
        x0 = self.encoder0(x)
        x0_att = self.SA0(x0)
        x0_side = self.conv_block_0(x0 * x0_att)

        x1 = self.encoder1(x0)
        x1_att = self.SA1(x1)
        x1_side = self.conv_block_1(x1 * x1_att)

        x2 = self.encoder2(x1)
        x2_att = self.SA2(x2)
        x2_side = self.conv_block_2(x2 * x2_att)

        x3 = self.encoder3(x2)
        x3_att = self.SA3(x3)
        x3_side = self.conv_block_3(x3 * x3_att)

        x4 = self.encoder4(x3)
        x4_att = self.SA4(x4)
        x4_side = self.conv_block_4(x4 * x4_att)

        x5 = self.pool(x4)
        x5 = x5.view(x5.size(0), -1)

        logit0 = self.fc0(x0_side)
        logit1 = self.fc1(x1_side)
        logit2 = self.fc2(x2_side)
        logit3 = self.fc3(x3_side)
        logit4 = self.fc4(x4_side)

        logit5 = self.fc5(x5)

        return logit0, logit1, logit2, logit3, logit4, logit5


    def get_fc_in_channels(self, x, block):
        with torch.no_grad():
            x = block(x)
            x = x.view(x.size(0), -1)
            return x.size(1)

    def get_conv_in_channels(self, x, block):
        with torch.no_grad():
            x = block(x)
            return x, x.size(1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DSResnetAttbranch(nn.Module):

    def __init__(self, pretrained=True, n_class=7):
        super(DSResnetAttbranch, self).__init__()

        # Resnet50
        # Blocks:
        layers = [3, 4, 6, 3]
        block = Bottleneck
        self.inplanes = 64

        params = {
            "pretrained": pretrained,
            "n_class": n_class
        }
        base_model = finetune_vggresnet(**params)

        kernel_size = 3
        out_channels = 64
        dropout = 0.2

        self.encoder0 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )

        self.att_feat0 = nn.Sequential(
            self._make_layer(block, 64, layers[0], stride=1, down_size=False),
            nn.BatchNorm2d(64 * block.expansion),
            nn.Conv2d(64 * block.expansion, n_class, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(n_class),
            nn.ReLU(),
        )

        self.att_soft0 = nn.Sequential(
            nn.Conv2d(n_class, n_class, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )

        self.att_sig0 = nn.Sequential(
            nn.Conv2d(n_class, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.encoder1 = base_model.layer1
        self._make_layer(block, 64, layers[0], down_size=True)

        self.att_feat1 = nn.Sequential(
            self._make_layer(block, 128, layers[1], stride=1, down_size=False),
            nn.BatchNorm2d(128 * block.expansion),
            nn.Conv2d(128 * block.expansion, n_class, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(n_class),
            nn.ReLU(),
        )

        self.att_soft1 = nn.Sequential(
            nn.Conv2d(n_class, n_class, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )

        self.att_sig1 = nn.Sequential(
            nn.Conv2d(n_class, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.encoder2 = base_model.layer2
        self._make_layer(block, 128, layers[1], stride=2, down_size=True)
        # self.inplanes = self.inplanes * block.expansion

        self.att_feat2 = nn.Sequential(
            self._make_layer(block, 256, layers[2], stride=1, down_size=False),
            nn.BatchNorm2d(256 * block.expansion),
            nn.Conv2d(256 * block.expansion, n_class, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(n_class),
            nn.ReLU(),
        )

        self.att_soft2 = nn.Sequential(
            nn.Conv2d(n_class, n_class, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )

        self.att_sig2 = nn.Sequential(
            nn.Conv2d(n_class, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.encoder3 = base_model.layer3
        self._make_layer(block, 256, layers[2], stride=2, down_size=True)

        self.att_feat3 = nn.Sequential(
            self._make_layer(block, 512, layers[3], stride=1, down_size=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.Conv2d(512 * block.expansion, n_class, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(n_class),
            nn.ReLU(),
        )

        self.att_soft3 = nn.Sequential(
            nn.Conv2d(n_class, n_class, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )

        self.att_sig3 = nn.Sequential(
            nn.Conv2d(n_class, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.encoder4 = base_model.layer4
        self.pool = base_model.avgpool

        # Last FC5
        self.fc5 = base_model.fc


    def forward(self, x):
        x = self.encoder0(x)
        att_feature_0 = self.att_feat0(x)
        att_sigmoid_0 = self.att_sig0(att_feature_0)
        att_softmax_0 = self.att_soft0(att_feature_0)
        x = x * att_sigmoid_0 + x

        x = self.encoder1(x)

        att_feature_1 = self.att_feat1(x)
        att_sigmoid_1 = self.att_sig1(att_feature_1)
        att_softmax_1 = self.att_soft1(att_feature_1)
        x = x * att_sigmoid_1 + x

        x = self.encoder2(x)

        att_feature_2 = self.att_feat2(x)
        att_sigmoid_2 = self.att_sig2(att_feature_2)
        att_softmax_2 = self.att_soft2(att_feature_2)
        x = x * att_sigmoid_2 + x

        x = self.encoder3(x)
        att_feature_3 = self.att_feat3(x)
        att_sigmoid_3 = self.att_sig3(att_feature_3)
        att_softmax_3 = self.att_soft3(att_feature_3)
        x = x * att_sigmoid_3 + x

        x = self.encoder4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)

        return att_softmax_0, att_softmax_1, att_softmax_2, att_softmax_3, x

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


    def _make_layer2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
