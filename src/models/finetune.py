import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model
from .abstract import ModelAbstract
from .vggface import vggface
from .vggresnet import vggresnet50
from .vggsenet import vggsenet50

from .irse import IR_50
import models.fishnet as fishnet


class Finetune(ModelAbstract):
    def __init__(
        self,
        arch="se_resnet50",
        n_class=6,
        pretrained=True,
        image_size=256,
    ):
        super(Finetune, self).__init__()
        self.model = make_model(
            model_name=arch,
            num_classes=n_class,
            pretrained=pretrained,
            input_size=(image_size, image_size),
        )

        self.extract_feature = False
        self.model_name = arch
        self.n_class = n_class

    def freeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.extract_feature:
            x = self.model._features(x)
            x = self.model.pool(x)
            x = x.view(x.size(0), -1)
            return x
        return self.model(x)

    def extract_features(self, x):
        x = self.model._features(x)
        x = self.model.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def reset_classifier(self):
        in_features = self.model._classifier.in_features
        self.model._classifier = nn.Linear(
            in_features=in_features,
            out_features=self.n_class
        )


class FinetuneVGGFace(ModelAbstract):
    def __init__(self, n_class=7, pretrained=True):
        super(FinetuneVGGFace, self).__init__()
        model = vggface(pretrained)
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv3 = model.conv3
        self.conv4 = model.conv4
        self.conv5 = model.conv5
        self.dropout = model.dropout
        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.fc3 = nn.Linear(4096, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def freeze_base(self):
        for module in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_base(self):
        for module in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2]:
            for param in module.parameters():
                param.requires_grad = True


class FinetuneVGGResnet(ModelAbstract):
    def __init__(self, n_class=7, pretrained=True, **kwargs):
        super(FinetuneVGGResnet, self).__init__()

        base_model = vggresnet50(pretrained=pretrained, num_classes=8631)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.fc = nn.Linear(2048, n_class)
        self.n_class = n_class

        self.extract_feature = False
        self.model_name = 'vggresnet'

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.extract_feature:
            return x

        x = self.fc(x)
        return x

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        return x

    def reset_classifier(self):
        in_features = self.fc.in_features
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=self.n_class
        )

    def freeze(self):
        for module in [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for module in [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool]:
            for param in module.parameters():
                param.requires_grad = True


class FinetuneVGGSenet(ModelAbstract):
    def __init__(self, n_class=7, pretrained=True, **kwargs):
        super(FinetuneVGGSenet, self).__init__()

        base_model = vggsenet50(pretrained=pretrained, num_classes=8631)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.fc = nn.Linear(2048, n_class)

        self.extract_feature = False
        self.model_name = 'vggsenet'

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.extract_feature:
            return x

        x = self.fc(x)
        return x

    def freeze_base(self):
        for module in [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_base(self):
        for module in [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool]:
            for param in module.parameters():
                param.requires_grad = True

    def reset_classifier(self):
        pass


class FinetuneIRModel(ModelAbstract):
    def __init__(self, n_class=7, pretrained=True):
        super(FinetuneIRModel, self).__init__()

        self.model = IR_50(pretrained=pretrained, input_size=[112, 112])

        self.fc = nn.Linear(512, n_class)

        self.extract_feature = False

    def forward(self, x):
        x = self.model(x)
        if self.extract_feature:
            return x

        x = self.fc(x)
        return x

    def freeze_base(self):
        for module in [self.model]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_base(self):
        for module in [self.model]:
            for param in module.parameters():
                param.requires_grad = True


class FinetuneFishNet(ModelAbstract):
    def __init__(self,
                 arch='fishnet99',
                 pretrained=None,
                 n_class=7
                 ):
        super(FinetuneFishNet, self).__init__()

        self.model = getattr(fishnet, arch)(pretrained=pretrained, n_class=n_class)
        self.model_name = arch
        self.extract_feature = False

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.pool1(x)
        score, score_feat = self.model.fish(x)
        # 1*1 output
        out = score.view(x.size(0), -1)
        if self.extract_feature:
            score_feat = F.adaptive_avg_pool2d(score_feat, 1)
            score_feat = score_feat.view(x.size(0), -1)
            return score_feat
        else:
            return out

    def extract_features(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.pool1(x)
        score, score_feat = self.model.fish(x)
        # 1*1 output
        score_feat = F.adaptive_avg_pool2d(score_feat, 1)
        score_feat = score_feat.view(x.size(0), -1)
        return score_feat

    def reset_classifier(self):
        pass

    def freeze_base(self):
        pass

    def unfreeze_base(self):
        pass


"""
Finetune the imagenet model
"""
def finetune(params):
    weight = params.get("backbone", None)
    if weight:
        params.pop('backbone')
    model = Finetune(**params)
    if weight:
        state_dict = torch.load(weight)
        model.load_state_dict(state_dict['model_state_dict'])
        print("Loaded pretrained model at {}".format(weight))
    return model


"""
Finetune VGG-Face
"""
def finetune_vggface(params):
    return FinetuneVGGFace(**params)


"""
Facetune resnet50 trained from VGGFace2
"""
def finetune_vggresnet(n_class=7, pretrained=None):
    model = FinetuneVGGResnet(n_class=n_class, pretrained=True)
    if pretrained:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict['model_state_dict'])
        print("Loaded pretrained model at {}".format(pretrained))
    return model


"""
Facetune senet50 trained from VGGFace2
"""
def finetune_vggsenet(params):
    weight = params.get("backbone", None)
    if weight:
        params.pop('backbone')
    model = FinetuneVGGSenet(**params)
    if weight:
        state_dict = torch.load(weight)
        model.load_state_dict(state_dict['model_state_dict'])
        print("Loaded pretrained model at {}".format(weight))
    return model



"""
Finetune IR model trained from Celeb
"""
def finetune_ir(params):
    weight = params.get("backbone", None)
    if weight:
        params.pop('backbone')
    model = FinetuneIRModel(**params)
    if weight:
        state_dict = torch.load(weight)
        model.load_state_dict(state_dict['model_state_dict'])
        print("Loaded pretrained model at {}".format(weight))
    return model


def finetune_fishnet(params):
    """
    Finetune fishmodel
    """
    return FinetuneFishNet(**params)