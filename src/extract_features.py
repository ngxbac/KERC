#!/usr/bin/env python
from collections import OrderedDict  # noqa F401
from torch.utils.data import DataLoader
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
import json
from dataset import FrameDataset
import models
from augmentation import valid_aug
import click


@click.group()
def cli():
    print("Extract features")


@cli.command()
@click.option('--model_name', type=str, default='vggresnet')
@click.option('--csv_file', type=str, default=None)
@click.option('--dataset', type=str, default='train')
@click.option('--feature_root', type=str)
def extract_features(
        model_name,
        csv_file,
        dataset,
        feature_root,
    ):
    log_dir = f"{feature_root}/{model_name}"
    with open(f"{log_dir}/config.json") as f:
        config = json.load(f)

    model_function = getattr(models, config['model_params']['model'])
    model = model_function(config['model_params']['params'])
    model.extract_feature = True

    loaders = OrderedDict()
    if csv_file:
        inferset = FrameDataset(
            csv_file=csv_file,
            transform=valid_aug(),
            mode='infer'
        )

        infer_loader = DataLoader(
            dataset=inferset,
            num_workers=4,
            shuffle=False,
            batch_size=16
        )

        loaders[f'infer_{dataset}'] = infer_loader

    checkpoint = f"{log_dir}/checkpoints/best.pth"
    callbacks = [
        CheckpointCallback(resume=checkpoint),
        InferCallback(out_dir=log_dir, out_prefix="/predicts_omg/predictions." + "{suffix}" + f".npy")
    ]
    runner = SupervisedRunner(
        input_key='image',
    )
    runner.infer(
        model,
        loaders,
        callbacks,
        verbose=True,
    )


if __name__ == "__main__":
    cli()
