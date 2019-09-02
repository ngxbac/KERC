import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import EmotiwPoolingFeature
from models import TemporalLSTM

import os
import numpy as np
import click
from tqdm import tqdm


@click.group()
def cli():
    print("Infer temporal models")


device = torch.device("cuda")


def get_feature_dims(model_name):
    n_features = 1024
    if model_name in [
        "densenet121"
    ]:
        n_features = 1024
    elif model_name in [
        "se_resnet50", "se_resnext50_32x4d",
        "inception_v3", "vggresnet",
        "resnet50"
    ]:
        n_features = 2048
    elif model_name in [
        "fishnet99"
    ]:
        n_features = 1056
    elif model_name in [
        "resnet18", "resnet34"
    ]:
        n_features = 512
    elif model_name in [
        "inception_v4", "inceptionresnetv2"
    ]:
        n_features = 1536

    return n_features


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = F.softmax(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


@cli.command()
@click.option('--log_dir', type=str)
@click.option('--model_name', type=str)
@click.option('--hidden_size', type=int)
@click.option('--feature_pkl', type=str)
@click.option('--out_dir', type=str)
def infer(
        log_dir,
        model_name,
        hidden_size,
        feature_pkl,
        out_dir
):
    n_features = get_feature_dims(model_name)
    model = TemporalLSTM(
        n_features=n_features,
        hidden_size=hidden_size,
        n_class=7
    )
    checkpoint = torch.load(
        f"{log_dir}/checkpoints/best.pth"
    )['model_state_dict']

    model.load_state_dict(checkpoint)
    model = model.to(device)
    dataset = EmotiwPoolingFeature(
        feature_pkl=feature_pkl,
        mode='train'
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    pred = predict(model, loader)

    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/predict.npy", pred)


if __name__ == '__main__':
    cli()

