import pandas as pd
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
import os
from tqdm import *

from models import *
from augmentation import *
from dataset import *

import click


device = torch.device('cuda')


@click.group()
def cli():
    print("Extract features from pretrained models")


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model.extract_features(images)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


def get_model_byname(model_name):
    if model_name == 'vggresnet':
        model = finetune_vggresnet(n_class=7)
    elif model_name == 'fishnet99':
        model = finetune_fishnet({
            "arch": model_name,
            "pretrained": None,
            "n_class": 7
        })
    else:
        model = finetune({
            "arch": model_name,
            "n_class": 7,
            "image_size": 256,
        })
    return model


@cli.command()
@click.option('--csv_file', type=str)
@click.option('--root', type=str)
@click.option('--model', type=str)
@click.option('--checkpoint', type=str)
@click.option('--save_feature_dir', type=str)
def extract(
    csv_file,
    root,
    model,
    checkpoint,
    save_feature_dir
):
    print("\n")
    print("****" * 50)
    print("CSV: ", csv_file)
    print("root: ", root)
    print("model: ", model)
    print("checkpoint: ", checkpoint)
    print("save_feature_dir: ", save_feature_dir)
    print("****" * 50)

    model = get_model_byname(model)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Dataset
    dataset = KERCDataset(
        csv_file=csv_file,
        transform=Compose([
            Resize(224, 224),
            Normalize(mean=MEAN_RGB, std=STD, max_pixel_value=1)
        ], p=1),
        mode='test',
        root=root
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    features = predict(model, loader)
    os.makedirs(save_feature_dir, exist_ok=True)
    np.save(save_feature_dir + "/features.npy", features)


if __name__ == '__main__':
    cli()
