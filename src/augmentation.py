from albumentations import *

input_range = [0, 1],
MEAN_RGB = [0.485, 0.456, 0.406],
STD = [0.229, 0.224, 0.225],


def train_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        HorizontalFlip(),
        Normalize(mean=MEAN_RGB, std=STD, max_pixel_value=255)
    ], p=1)


def train_sfew_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        Rotate(10),
        HorizontalFlip(),
        Normalize(mean=MEAN_RGB, std=STD, max_pixel_value=255)
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        Normalize(mean=MEAN_RGB, std=STD, max_pixel_value=255)
    ], p=1)


def valid_sfew_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        Normalize(mean=MEAN_RGB, std=STD, max_pixel_value=255)
    ], p=1)