# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from losses import *
from callbacks import *
from optimizers import *
from schedulers import *


# Register models
registry.Model(finetune_vggresnet)
registry.Model(DeepBranchResnet)
registry.Model(DeepBranchAttResnet)
registry.Model(DSInceptionV3)
registry.Model(DSResnet)
registry.Model(SSEResnet)
registry.Model(DSResnetSA)
registry.Model(DSResnetAttbranch)
registry.Model(GAIN)
registry.Model(GCAM)
registry.Model(GAINMask)

# Register callbacks
registry.Callback(LabelSmoothCriterionCallback)
registry.Callback(SmoothMixupCallback)
registry.Callback(DSAccuracyCallback)
registry.Callback(DSCriterionCallback)
registry.Callback(GAINCriterionCallback)
registry.Callback(GAINSaveHeatmapCallback)
registry.Callback(GCAMSaveHeatmapCallback)
registry.Callback(GAINMaskCriterionCallback)

# Register criterions
registry.Criterion(LabelSmoothingCrossEntropy)

# Register optimizers
registry.Optimizer(AdamW)
registry.Optimizer(Nadam)

registry.Scheduler(CyclicLRFix)