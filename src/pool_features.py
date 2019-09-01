import os
import pandas as pd
import numpy as np
import pickle
import click


@click.group()
def cli():
    print("Pooling features")


@cli.command()
@click.option('--features_path', type=str, default='vggresnet')
@click.option('--csv_file', type=str, default=None)
def pool_features(
        features_path,
        csv_file,
):
    features = np.load(f"{features_path}/features.npy")
    df = pd.read_csv(csv_file)

    assert df.shape[0] == features.shape[0]

    test_video_features = pooling_dataset(df, features, stride=2, pool_out=8)

    with open(f"{features_path}/pooled_features.pkl", 'wb') as f:
        pickle.dump(test_video_features, f)


def pad_if_need(paths, size):
    diff = len(paths) - size

    if diff < 0:
        if abs(diff) > len(paths):
            up_sampling = paths[np.random.choice(paths.shape[0], abs(diff), replace=True)]
        else:
            up_sampling = paths[np.random.choice(paths.shape[0], abs(diff), replace=False)]
        paths = np.concatenate([paths, up_sampling])

    return paths


def pooling_video(video_df: pd.DataFrame, video_features: np.array, stride=2, pool_out=16):
    video_df['frame_num'] = video_df['frame_name'].apply(lambda x: int(x.split('.')[0].split('_')[-1]))
    video_df = video_df.sort_values(by='frame_num')
    video_features = video_features[video_df.index]

    pooled_features = video_features

    while len(pooled_features) > pool_out:
        i = 0
        temp_pooled_features = []
        while i <= len(pooled_features) - stride:
            max_feature = np.max(pooled_features[i:i+stride], axis=0)
            temp_pooled_features.append(max_feature)
            i += stride
        temp_pooled_features = np.asarray(temp_pooled_features)
        pooled_features = temp_pooled_features

    pooled_features = pad_if_need(pooled_features, pool_out)

    return pooled_features


def pooling_dataset(df: pd.DataFrame, features: np.array, stride=2, pool_out=16):
    all_video = df['video_name'].unique()

    all_video_features = {}
    for video in all_video:
        video_df = df[df['video_name'] == video]
        video_features = features[video_df.index]
        video_df = video_df.reset_index(drop=True)
        emotion = video_df['emotion'].unique()[0]

        pooled_feature = pooling_video(video_df, video_features, stride, pool_out)

        all_video_features[video] = {}
        all_video_features[video]['feature'] = pooled_feature
        all_video_features[video]['emotion'] = emotion

    return all_video_features


if __name__ == '__main__':
    cli()
