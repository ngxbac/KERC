import pandas as pd
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
import os
from tqdm import *

from models import *
from augmentation import *
from dataset import FrameDataset


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = Ftorch.softmax(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


def predict_all():
    test_csv = "./notebooks/valid.csv.gz"
    model_name = 'vggresnet_ds'
    log_dir = f"/media/ngxbac/DATA/logs_omg/{model_name}/learnable_weight/"

    model = DeepBranchResnet(
        n_class=7,
        pretrained="/media/ngxbac/DATA/logs_emotiw_temporal/feature_extractor/vggresnet/checkpoints/best.pth"
    )

    checkpoint = f"{log_dir}/checkpoints/best.pth"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Dataset
    dataset = FrameDataset(
        csv_file=test_csv,
        transform=valid_aug(224),
        mode='test'
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    pred = predict(model, loader)

    # pred = np.asarray(pred).mean(axis=0)
    all_preds = np.argmax(pred, axis=1)
    df = pd.read_csv(test_csv)
    submission = df.copy()
    submission['EmotionMaxVote'] = all_preds.astype(int)
    os.makedirs("prediction", exist_ok=True)
    submission.to_csv(f'./prediction/{model_name}.csv', index=False, columns=['video', 'utterance', 'EmotionMaxVote'])
    np.save(f"./prediction/{model_name}.npy", pred)


if __name__ == '__main__':
    predict_all()
