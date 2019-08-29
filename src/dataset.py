import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(path):
    image = cv2.imread(path, 0)
    return image


class FrameDataset(Dataset):
    def __init__(self, csv_file, transform=None, mode="train"):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            df = pd.read_csv(csv_file, nrows=None)

        self.paths = df["file"].values.tolist()
        self.labels = df["emotion"].values.tolist()
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        image = load_image(path)
        if self.mode == "train":
            label = self.labels[idx]
        else:
            label = -1

        if self.transform:
            data = {"image": image}
            augmented = self.transform(**data)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label
        }


class AffectNetDataset(Dataset):
    def __init__(self, root, df_path, transform=None, mode="train"):
        self.root = root
        df = pd.read_csv(df_path, nrows=None)

        self.paths = df["subDirectory_filePath"].values.tolist()
        le = LabelEncoder()
        self.labels = le.fit_transform(df["emotion"].values)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        path = os.path.join(self.root, path)

        image = load_image(path)
        if self.mode == "train":
            label = self.labels[idx]
        else:
            label = -1

        if self.transform:
            data = {"image": image}
            augmented = self.transform(**data)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label
        }


class RAFDataset(Dataset):
    def __init__(self, df_path, transform=None, mode="train"):
        df = pd.read_csv(df_path, nrows=None)
        self.paths = df["path"].values.tolist()
        self.labels = df["label"].values.tolist()
        self.transform = transform
        self.mode = mode
        self.root_mask = "/media/ngxbac/Bac/dataset/RAFDatabase/basic/Image/mask_select/"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image_name = path.split("/")[-1]
        mask_path = os.path.join(self.root_mask, image_name)
        image = load_image(path)
        mask = load_mask(mask_path)
        if mask is None:
            mask = np.zeros_like((image.shape[0], image.shape[1]))
        else:
            mask = (mask / 255).astype(np.uint8)
        if self.mode == "train":
            label = self.labels[idx] - 1
        else:
            label = -1

        if self.transform:
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image = augmented["image"]
            mask = augmented["mask"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask, 0).astype(np.float32)

        # print(image.shape)
        return {
            "images": image,
            "targets": label,
            "masks": mask,
            "image_names": image_name
        }


class SFEWDataset(Dataset):
    def __init__(self,
                 df_path,
                 root,
                 root_mask,
                 transform=None,
                 mode='train',
                 ):
        df = pd.read_csv(df_path, nrows=None)
        self.images = df['image'].values
        self.emotions = df['emotion'].values
        self.labels = LabelEncoder().fit_transform(self.emotions)
        self.root = root
        self.root_mask = root_mask
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        emotion = self.emotions[idx]
        image_path = os.path.join(self.root, emotion, image_name)
        mask_path = os.path.join(self.root_mask, image_name)
        image = load_image(image_path)
        mask = load_mask(mask_path)
        if mask is None:
            mask = np.zeros_like((image.shape[0], image.shape[1]))
        else:
            mask = (mask / 255).astype(np.uint8)

        if self.mode == "train":
            label = self.labels[idx]
        else:
            label = -1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask, 0).astype(np.float32)

        return {
            "images": image,
            "targets": label,
            "masks": mask,
            "image_names": image_name
        }


class KERCDataset(Dataset):
    def __init__(self, csv_file, root, transform=None, mode="train"):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            df = pd.read_csv(csv_file, nrows=None)

        df['file'] = df['file'].apply(lambda x: "/".join(x.split("/")[-4:]))
        self.paths = df["file"].values.tolist()
        self.labels = LabelEncoder().fit_transform(df["emotion"].values)
        self.root = root
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        path = os.path.join(self.root, path)

        image = load_image(path)
        if self.mode == "train":
            label = self.labels[idx]
        else:
            label = -1

        if self.transform:
            data = {"image": image}
            augmented = self.transform(**data)
            image = augmented["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label
        }