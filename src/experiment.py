from collections import OrderedDict
import torch
import torch.nn as nn
import random
from catalyst.dl.experiment import ConfigExperiment
from dataset import *
from augmentation import train_aug, valid_aug, train_sfew_aug, valid_sfew_aug


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings("ignore")

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage == "stage0":
            if hasattr(model_, 'freeze'):
                model_.freeze()
                print("Freeze backbone model !!!")
            else:
                for param in model_._features.parameters():
                    param.requires_grad = False
                print("Freeze backbone model !!!")

        else:
            if hasattr(model_, 'unfreeze'):
                model_.unfreeze()
                print("Unfreeze backbone model !!!")
            else:
                for param in model_._features.parameters():
                    param.requires_grad = True
                print("Freeze backbone model !!!")

        return model_

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        """
        image_key: 'id'
        label_key: 'attribute_ids'
        """

        image_size = kwargs.get("image_size", 320)
        train_csv = kwargs.get('train_csv', None)
        valid_csv = kwargs.get('valid_csv', None)
        root = kwargs.get('root', None)

        if train_csv:
            transform = train_aug(image_size)
            train_set = KERCDataset(
                csv_file=train_csv,
                transform=transform,
                mode='train',
                root=root
            )
            datasets["train"] = train_set

        if valid_csv:
            transform = valid_aug(image_size)
            valid_set = KERCDataset(
                csv_file=valid_csv,
                transform=transform,
                mode='train',
                root=root
            )
            datasets["valid"] = valid_set

        affectnet_train_csv = kwargs.get("affectnet_train_csv", None)
        affectnet_valid_csv = kwargs.get("affectnet_valid_csv", None)
        affectnet_root = kwargs.get("affectnet_root", None)

        if affectnet_train_csv is not None:

            train_dataset = AffectNetDataset(
                root=affectnet_root,
                df_path=affectnet_train_csv,
                transform=train_aug(image_size),
                mode="train"
            )
            datasets["train"] = train_dataset

        if affectnet_valid_csv is not None:
            valid_dataset = AffectNetDataset(
                root=affectnet_root,
                df_path=affectnet_valid_csv,
                transform=valid_aug(image_size),
                mode="train"
            )
            datasets["valid"] = valid_dataset

        """
        RAF Database
        """
        raf_train_csv = kwargs.get("raf_train_csv", None)
        raf_valid_csv = kwargs.get("raf_valid_csv", None)

        if raf_train_csv is not None:
            train_dataset = RAFDataset(
                raf_train_csv,
                transform=train_aug(image_size),
                mode="train"
            )
            datasets["train"] = train_dataset

        if raf_valid_csv is not None:
            valid_dataset = RAFDataset(
                raf_valid_csv,
                # transform=Experiment.get_transforms(stage=stage, mode='valid'),
                transform=valid_aug(image_size),
                mode="train"
            )
            datasets["valid"] = valid_dataset


        """
        SFEW Database
        """
        sfew_train_csv = kwargs.get("sfew_train_csv", None)
        sfew_valid_csv = kwargs.get("sfew_valid_csv", None)
        sfew_train_root_image = kwargs.get("sfew_train_root_image", None)
        sfew_train_root_mask = kwargs.get("sfew_train_root_mask", None)

        sfew_valid_root_image = kwargs.get("sfew_valid_root_image", None)
        sfew_valid_root_mask = kwargs.get("sfew_valid_root_mask", None)

        if sfew_train_csv is not None:
            train_dataset = SFEWDataset(
                sfew_train_csv,
                root=sfew_train_root_image,
                root_mask=sfew_train_root_mask,
                transform=train_sfew_aug(image_size),
                mode="train"
            )
            datasets["train"] = train_dataset

        if sfew_valid_csv is not None:
            valid_dataset = SFEWDataset(
                sfew_valid_csv,
                root=sfew_valid_root_image,
                root_mask=sfew_valid_root_mask,
                transform=valid_sfew_aug(image_size),
                mode="train"
            )
            datasets["valid"] = valid_dataset

        """
        Temporal dataset
        """
        train_pool = kwargs.get("train_pool", None)
        valid_pool = kwargs.get("valid_pool", None)

        if train_pool is not None:
            train_dataset = EmotiwPoolingFeature(
                feature_pkl=train_pool,
                mode="train"
            )
            datasets["train"] = train_dataset

        if valid_pool is not None:
            valid_dataset = EmotiwPoolingFeature(
                feature_pkl=valid_pool,
                mode="train"
            )
            datasets["valid"] = valid_dataset

        return datasets


