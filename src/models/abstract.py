import torch.nn as nn


class ModelAbstract(nn.Module):
    def freeze_base(self):
        pass

    def unfreeze_base(self):
        pass
