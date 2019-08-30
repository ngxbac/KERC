import os
import pandas as pd
import numpy as np
from tqdm import *
import click

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV


@click.group()
def cli():
    print("Search parameters")


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


class Objective(object):
    def __init__(self, classifier, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train

        self.X_valid = X_valid
        self.y_valid = y_valid

        self.classifier = classifier

    def get_classifier(self, classifier, trial):
        if classifier == 'svm':
            svc_c = trial.suggest_loguniform('svc_c', 5e-4, 1)
            clf = SVC(C=svc_c, kernel='linear')

        return clf

    def __call__(self, trial):
        clf = self.get_classifier(self.classifier, trial)
        clf.fit(self.X_train, self.y_train)
        accuracy = clf.score(self.X_valid, self.y_valid)
        return accuracy


@cli.command()
@click.option('--feature_dir', type=str)
@click.option('--train_csv', type=str)
@click.option('--valid_csv', type=str)
@click.option('--classifier', type=str, default='svm')
@click.option('--out_config', type=str, default='ml_configs')
@click.option('--n_trials', type=int, default=100)
@click.option('--feature_name', type=str)
def search_svm(
    feature_dir,
    train_csv,
    valid_csv,
    classifier,
    out_config,
    n_trials,
    feature_name,
):
    os.makedirs(f'{out_config}/{feature_name}/{classifier}', exist_ok=True)
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

    objective = Objective(
        X_train=X_video_train,
        y_train=y_video_train,
        X_valid=X_video_valid,
        y_valid=y_video_valid,
        classifier=classifier
    )

    import optuna
    import json
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    with open(f'{out_config}/{feature_name}/{classifier}/config.json', "w") as f:
        json.dump(study.best_trial.params, f)


if __name__ == '__main__':
    cli()


