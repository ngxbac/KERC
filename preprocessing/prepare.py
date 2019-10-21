import sys
sys.path.append('/workspace/mtcnn-pytorch/')
import pandas as pd
import numpy as np
import os
import glob
import click
import cv2
from tqdm import tqdm
from src import detect_faces
from PIL import Image


# @click.group()
# def cli():
#     print("Preprocessing for KERC")


def get_emotion(path):
    return path.split("/")[-2]


def get_video_name(path):
    return path.split("/")[-1]


def get_dataset(path):
    return path.split("/")[-3]


def video_to_frame(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    while success:
        frames.append(image)
        success, image = vidcap.read()

    return frames


def video_df_to_frame(df, frame_dir):
    frame_files = []
    video_names = []
    emotions = []
    for file, emotion, video_name in tqdm(zip(
        df.file, df.emotion, df.video_name
    ), total=len(df)):
        out_frame_dir = os.path.join(frame_dir, emotion, video_name)
        os.makedirs(out_frame_dir, exist_ok=True)
        frames = video_to_frame(file)
        for i, frame in enumerate(frames):
            out_file = os.path.join(out_frame_dir, f'frame_{i}.jpg')
            cv2.imwrite(out_file, frame)

            frame_files.append(out_file)
            video_names.append(video_name)
            emotions.append(emotion)
    df = pd.DataFrame({
        'file': frame_files,
        'video_name': video_names,
        'emotion': emotions
    })

    return df


def frame_to_face(frame_path, detect_multiple_faces=True, margin=0):
    img = Image.open(frame_path)
    bounding_boxes, _ = detect_faces(
        img,
        pretrain_pnet='/workspace/mtcnn-pytorch/src/weights/pnet.npy',
        pretrain_rnet='/workspace/mtcnn-pytorch/src/weights/rnet.npy',
        pretrain_onet='/workspace/mtcnn-pytorch/src/weights/onet.npy'
    )

    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_faces = []
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = img.shape[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cropped
            all_faces.append(scaled)

    return all_faces


def frame_df_to_face(df, face_dir):
    face_files = []
    frame_names = []
    video_names = []
    emotions = []

    import pdb
    pdb.set_trace()
    for file, emotion, video_name in tqdm(zip(
        df.file, df.emotion, df.video_name
    ), total=len(df)):
        out_face_dir = os.path.join(face_dir, emotion, video_name)
        frame_name = file.split("/")[-1]
        os.makedirs(out_face_dir, exist_ok=True)
        if len(glob.glob(f"{out_face_dir}/{frame_name}*.jpg")) != 0:
            continue

        try:
            faces = frame_to_face(file)
            for i, face in enumerate(faces):
                out_file = os.path.join(out_face_dir, f'{frame_name}_{i}.jpg')
                cv2.imwrite(out_file, face)

                face_files.append(out_file)
                video_names.append(video_name)
                emotions.append(emotion)
                frame_names.append(frame_name)
        except:
            print(f"{file}")

    df = pd.DataFrame({
        'file': face_files,
        'frame_name': frame_names,
        'video_name': video_names,
        'emotion': emotions
    })

    return df


def video_csv(dir):
    videos = glob.glob(f"{dir}/*/*.mp4")
    df = pd.DataFrame()
    df['file'] = videos
    df['emotion'] = df['file'].apply(lambda x: get_emotion(x))
    df['dataset'] = df['file'].apply(lambda x: get_dataset(x))
    df['video_name'] = df['file'].apply(lambda x: get_video_name(x))
    return df


def preprocess():
    train_video_dir = "/media/ngxbac/Bac/dataset/KERC/video/train/"
    valid_video_dir = "/media/ngxbac/Bac/dataset/KERC/video/val/"
    test_video_dir = "/media/ngxbac/Bac/dataset/KERC/video/test/"

    train_frame_dir = "/media/ngxbac/Bac/dataset/KERC/frames/train/"
    valid_frame_dir = "/media/ngxbac/Bac/dataset/KERC/frames/val/"
    test_frame_dir = "/media/ngxbac/Bac/dataset/KERC/frames/test/"

    train_face_dir = "/media/ngxbac/Bac/dataset/KERC/faces/train/"
    valid_face_dir = "/media/ngxbac/Bac/dataset/KERC/faces/val/"
    test_face_dir = "/media/ngxbac/Bac/dataset/KERC/faces/test/"

    csv_dir = "./csv/"
    os.makedirs(csv_dir, exist_ok=True)

    # if train_video_dir:
    #     # train_df = video_csv(train_video_dir)
    #     # train_df.to_csv(f"{csv_dir}/train_video.csv", index=False)
    #     # # Extract video to frames
    #     # os.makedirs(train_frame_dir, exist_ok=True)
    #     # train_frame_df = video_df_to_frame(train_df, train_frame_dir)
    #     # train_frame_df.to_csv(f"{csv_dir}/train_frame.csv", index=False)
    #     train_frame_df = pd.read_csv(f"{csv_dir}/train_frame.csv")
    #     # Extract detect face from frames
    #     train_face_df = frame_df_to_face(train_frame_df, train_face_dir)
    #     train_face_df.to_csv(f"{csv_dir}/train_face.csv", index=False)
    #
    # if valid_video_dir:
    #     valid_df = video_csv(valid_video_dir)
    #     valid_df.to_csv(f"{csv_dir}/valid_video.csv", index=False)
    #     # Extract video to frames
    #     os.makedirs(valid_frame_dir, exist_ok=True)
    #     valid_frame_df = video_df_to_frame(valid_df, valid_frame_dir)
    #     valid_frame_df.to_csv(f"{csv_dir}/valid_frame.csv", index=False)
    #     # Extract detect face from frames
    #     valid_face_df = frame_df_to_face(valid_frame_df, valid_face_dir)
    #     valid_face_df.to_csv(f"{csv_dir}/valid_face.csv", index=False)

    if test_video_dir:
        # test_df = video_csv(test_video_dir)
        # overlap_df = pd.read_csv("./preprocessing/val_test_map.csv")
        # test_df = test_df[~test_df['file'].isin(overlap_df['path_test'])].reset_index(drop=True)
        # test_df.to_csv(f"{csv_dir}/test_video.csv", index=False)

        # test_df = pd.read_csv(f"{csv_dir}/test_video.csv")
        # Extract video to frames
        os.makedirs(test_frame_dir, exist_ok=True)
        # test_frame_df = video_df_to_frame(test_df, test_frame_dir)
        # test_frame_df.to_csv(f"{csv_dir}/test_frame.csv", index=False)

        test_frame_df = pd.read_csv(f"{csv_dir}/test_frame.csv")
        # Extract detect face from frames
        test_face_df = frame_df_to_face(test_frame_df, test_face_dir)
        test_face_df.to_csv(f"{csv_dir}/test_face.csv", index=False)


if __name__ == '__main__':
    preprocess()
