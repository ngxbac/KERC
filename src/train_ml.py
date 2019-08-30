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
    print("Train KERC with SVM")


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


def train_svm_calib(X_train, y_train, X_valid, y_valid):
    SVM_C_range = [5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
    print('Cross-validation')
    cv_acc = []
    for SVM_C in SVM_C_range:
        print('C = %f' % SVM_C)

        # Fix random_state, otherwise not reproducable
        cv = StratifiedKFold(5, shuffle=True, random_state=4096)

        clf = make_pipeline(StandardScaler(), LinearSVC(C=SVM_C))
        scores = cross_val_score(clf, X_train, y_train, cv=cv)
        cv_acc.append(np.mean(scores))
        print(cv_acc[-1])

    ind = np.argmax(cv_acc)
    SVM_C = SVM_C_range[ind]
    print('best_C = %f' % SVM_C)
    clf = make_pipeline(StandardScaler(), LinearSVC(C=SVM_C))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    train_acc = 100*np.mean(np.equal(np.array(pred_train), y_train))
    print('train acc = %f' % train_acc)
    pred = clf.predict(X_valid)
    decision_values = clf.decision_function(X_valid)
    val_acc = 100*np.mean(np.equal(np.array(pred), y_valid))
    print('val acc = %f' % val_acc)
    return train_acc, val_acc, SVM_C, decision_values, clf, pred


@cli.command()
@click.option('--feature_dir', type=str)
@click.option('--train_csv', type=str)
@click.option('--valid_csv', type=str)
def train_svm(
    feature_dir,
    train_csv,
    valid_csv
):
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

    model = make_pipeline(StandardScaler(), SVC(probability=True, kernel='linear'))
    # model = CalibratedClassifierCV(model, cv=5)
    model.fit(X_video_train, y_video_train)
    y_pred = model.predict_proba(X_video_valid)
    y_pred = np.argmax(y_pred, axis=1)
    print(accuracy_score(y_video_valid, y_pred))


if __name__ == '__main__':
    cli()


