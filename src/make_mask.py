import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from joblib import delayed, Parallel
from tqdm import *


landmark_dir = "/media/ngxbac/Bac/dataset/RAFDatabase/basic/Image/landmark/"

emotion_dict = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}


def iou(img1, img2):
    overlap = img1 * img2  # Logical AND
    union = img1 + img2  # Logical OR
    return overlap.sum() / (float(union.sum()) + 1e-17)


def make_mask_face_expand(raw_image, landmark):
    """
    Generate landmark of entry face which does not have nose
    """
    face_bound = list(range(0, 27))
    uper_bound = list(range(17, 27))
    nose_bound = list(range(27, 36))

    shape = landmark

    mask = np.zeros_like(raw_image)
    uper_bound_contours = shape[uper_bound]
    uper_bound_contours[:, 1] = uper_bound_contours[:, 1] - 10
    shape[uper_bound] = uper_bound_contours
    face_contours = shape[face_bound].astype(np.int)
    face_contours = cv2.convexHull(face_contours, False).reshape(-1, 2)
    cv2.fillPoly(mask, pts=[face_contours], color=(1, 1, 1))
    return mask


def get_landmark_mask(raw_image, landmark, label):
    """
    Generate landmark mask which has eyes + mouth
    """
    left_eye = list(range(36, 41 + 1))
    right_eye = list(range(42, 47 + 1))
    mouth = list(range(48, 59 + 1))

    shape = landmark

    left_eye_contours = shape[left_eye].astype(np.int)
    right_eye_contours = shape[right_eye].astype(np.int)
    mouth_contours = shape[mouth].astype(np.int)

    left_eye = False
    right_eye = False
    mouth = False

    only_left_eye = np.zeros_like(raw_image)
    cv2.fillPoly(only_left_eye, pts=[left_eye_contours], color=(1, 1, 1))

    only_right_eye = np.zeros_like(raw_image)
    cv2.fillPoly(only_right_eye, pts=[right_eye_contours], color=(1, 1, 1))

    only_mouth = np.zeros_like(raw_image)
    cv2.fillPoly(only_mouth, pts=[mouth_contours], color=(1, 1, 1))

    right_mouth_iou = iou(only_right_eye, only_mouth)
    left_mouth_iou = iou(only_left_eye, only_mouth)
    right_left_iou = iou(only_left_eye, only_right_eye)

    if right_mouth_iou + left_mouth_iou + right_left_iou == 0:
        left_eye = True
        right_eye = True
        mouth = True
    else:
        if right_mouth_iou == 0:
            right_eye = True
            mouth = True
            left_eye = False

        if left_mouth_iou == 0:
            left_eye = True
            right_eye = False
            mouth = True

        if right_left_iou == 0:
            left_eye = True
            right_eye = True
            mouth = False

    left_eye_contours[[0, 1, 2, 3], 1] = left_eye_contours[[0, 1, 2, 3], 1] - 10
    left_eye_contours[[4, 5], 1] = left_eye_contours[[4, 5], 1] + 10

    right_eye_contours[[0, 1, 2, 3], 1] = right_eye_contours[[0, 1, 2, 3], 1] - 10
    right_eye_contours[[4, 5], 1] = right_eye_contours[[4, 5], 1] + 10

    mouth_contours[[0, 1, 2, 3, 4, 5, 6], 1] = mouth_contours[[0, 1, 2, 3, 4, 5, 6], 1] - 10
    mouth_contours[[7, 8, 9, 10, 11], 1] = mouth_contours[[7, 8, 9, 10, 11], 1] + 10

    """
    All face: Neutral, Fear
    Face components: Happy, surprise
    """
    emotion = emotion_dict[label]

    mask = None

    if left_eye + right_eye + mouth != 0:
        if emotion in ["Neutral", "Fear"]:
            mask = make_mask_face_expand(raw_image, landmark)

        if emotion in ["Happiness", "Surprise"]:
            mask = np.zeros_like(raw_image)

            if left_eye:
                cv2.fillPoly(mask, pts=[left_eye_contours], color=(1, 1, 1))
            if right_eye:
                cv2.fillPoly(mask, pts=[right_eye_contours], color=(1, 1, 1))
            if mouth:
                cv2.fillPoly(mask, pts=[mouth_contours], color=(1, 1, 1))

    return mask


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_landmark(image_name):
    landmark = np.load(os.path.join(landmark_dir, image_name + ".npy"))
    return landmark


def parallel_make_mask(path, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    image_name = path.split("/")[-1]
    outfile = os.path.join(out_dir, image_name)
    image = load_image(path)
    landmark = get_landmark(image_name)
    mask = get_landmark_mask(image, landmark, label)
    if not (mask is None):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8) * 255
        cv2.imwrite(outfile, mask)


if __name__ == '__main__':
    df = pd.read_csv("/media/ngxbac/Bac/competition/emotiw/notebook/RAF/csv/train.csv")
    out_dir = "/media/ngxbac/Bac/dataset/RAFDatabase/basic/Image/mask_select/"
    Parallel(n_jobs=4)(
        delayed(parallel_make_mask)(path, out_dir, label) for \
        path, label in tqdm(zip(df.path.values, df.label.values))
    )
