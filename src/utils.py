import pandas as pd
import numpy as np


from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


FEATURE_FUNCS = {
    'mean': np.mean,
    'min': np.min,
    'max': np.max,
    'std': np.std,
    'skew': skew,
    'kurtosis': kurtosis,
    'median': np.median
}


def feature_engineering(X, y, df, feature_funcs):
    features = []
    labels = []

    videos = df['video_name'].unique()
    for video in videos:
        video_df = df[df['video_name'] == video]
        X_video = X[video_df.index]
        y_video = y[video_df.index]

        all_features = [FEATURE_FUNCS[func](X_video, axis=0).reshape(1, -1) for func in feature_funcs]
        feature_concat = np.concatenate(all_features, axis=1)

        features.append(feature_concat)
        labels.append(y_video[0])

    features = np.concatenate(features, axis=0)
    return features, np.asarray(labels)


def get_data(train_csv, valid_csv, feature_dir):
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    le = LabelEncoder()
    train_df['emotion'] = le.fit_transform(train_df['emotion'])
    valid_df['emotion'] = le.transform(valid_df['emotion'])

    X_train = np.load(f"{feature_dir}/train/features.npy")
    X_valid = np.load(f"{feature_dir}/valid/features.npy")

    y_train = train_df['emotion'].values
    y_valid = valid_df['emotion'].values

    assert X_train.shape[0] == train_df.shape[0]
    assert X_valid.shape[0] == valid_df.shape[0]

    feature_functions = ['mean', 'max', 'std']

    X_video_train, y_video_train = feature_engineering(
        X=X_train, y=y_train, df=train_df,
        feature_funcs=['mean', 'max', 'std']
    )

    X_video_valid, y_video_valid = feature_engineering(
        X=X_valid, y=y_valid, df=valid_df,
        feature_funcs=['mean', 'max', 'std']
    )

    n_features = len(feature_functions)
    feature_dims = X_video_train.shape[1] // n_features
    print("Feature dims: ", feature_dims)
    for i in range(n_features):
        scaler = MinMaxScaler()
        X_video_train[:, i:i + feature_dims] = scaler.fit_transform(X_video_train[:, i:i + feature_dims])
        X_video_valid[:, i:i + feature_dims] = scaler.transform(X_video_valid[:, i:i + feature_dims])

    return X_video_train, X_video_valid, y_video_train, y_video_valid
