import os
import pandas as pd
import numpy as np
from tqdm import *
import click

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
import json
import pickle


@click.group()
def cli():
    print("Train KERC with sklearn")


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


def load_config(config_dir, classifier, feature_name):
    with open(f'{config_dir}/{feature_name}/{classifier}/config.json', "r") as f:
        config = json.load(f)

    return config


def save_model(save_dir, classifier, feature_name, fold, model):
    if fold is not None:
        with open(f'{save_dir}/{feature_name}/{classifier}/model_{fold}.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(f'{save_dir}/{feature_name}/{classifier}/model.pkl', 'wb') as f:
            pickle.dump(model, f)


def save_prediction(save_dir, classifier, feature_name, fold, pred):
    if fold is not None:
        np.save(f'{save_dir}/{feature_name}/{classifier}/valid_{fold}.npy', pred)
    else:
        np.save(f'{save_dir}/{feature_name}/{classifier}/valid.npy', pred)


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


@cli.command()
@click.option('--feature_dir', type=str)
@click.option('--train_csv', type=str)
@click.option('--valid_csv', type=str)
@click.option('--classifier', type=str, default='svm')
@click.option('--config_dir', type=str, default='ml_configs')
@click.option('--feature_name', type=str)
@click.option('--save_dir', type=str)
def train_normal(
    feature_dir,
    train_csv,
    valid_csv,
    config_dir,
    classifier,
    feature_name,
    save_dir,
):
    X_video_train, X_video_valid, y_video_train, y_video_valid = get_data(train_csv, valid_csv, feature_dir)

    config = load_config(config_dir, classifier, feature_name)
    # model = SVC(kernel='linear', probability=True, C=config['svc_c'])
    if classifier == 'svm':
        model = SVC(kernel='linear', probability=True, C=config['svc_c'])
    elif classifier == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=config['n_estimators'])
    else:
        raise ValueError("Not support classifier: {}".format(classifier))
    model.fit(X_video_train, y_video_train)
    y_pred = model.predict_proba(X_video_valid)
    y_pred_cls = np.argmax(y_pred, axis=1)
    print(feature_name)
    print("Accuracy ", accuracy_score(y_video_valid, y_pred_cls))

    # Save model and valid's prediction
    os.makedirs(f'{save_dir}/{feature_name}/{classifier}/', exist_ok=True)
    save_model(save_dir, classifier, feature_name, None, model)
    save_prediction(save_dir, classifier, feature_name, None, y_pred)


@cli.command()
@click.option('--feature_dir', type=str)
@click.option('--train_csv', type=str)
@click.option('--valid_csv', type=str)
@click.option('--classifier', type=str, default='svm')
@click.option('--config_dir', type=str, default='ml_configs')
@click.option('--feature_name', type=str)
@click.option('--save_dir', type=str)
@click.option('--kfold_idx_dir', type=str)
def train_kfold(
    feature_dir,
    train_csv,
    valid_csv,
    config_dir,
    classifier,
    feature_name,
    save_dir,
    kfold_idx_dir,
):
    print("Classifier: {}".format(classifier))
    X_video_train, X_video_valid, y_video_train, y_video_valid = get_data(train_csv, valid_csv, feature_dir)

    X = np.concatenate([X_video_train, X_video_valid], axis=0)
    y = np.concatenate([y_video_train, y_video_valid], axis=0)

    kf = StratifiedKFold(n_splits=5, random_state=2411, shuffle=True)
    oof_pred = np.zeros_like(y)
    # for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    for fold in range(5):
        train_idx = np.load(f"./{kfold_idx_dir}/train_{fold}.npy")
        valid_idx = np.load(f"./{kfold_idx_dir}/valid_{fold}.npy")

        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

        config = load_config(config_dir, classifier, feature_name)
        if classifier == 'svm':
            model = SVC(kernel='linear', probability=True, C=config['svc_c'])
        elif classifier == 'RandomForestClassifier':
            model = RandomForestClassifier(n_estimators=config['n_estimators'])
        else:
            raise ValueError("Not support classifier: {}".format(classifier))
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict_proba(X_valid_fold)
        y_pred_cls = np.argmax(y_pred, axis=1)
        print(feature_name)
        print("Fold {}, Accuracy {}".format(fold, accuracy_score(y_valid_fold, y_pred_cls)))

        oof_pred[valid_idx] = y_pred_cls

        # Save model and valid's prediction
        os.makedirs(f'{save_dir}/{feature_name}/{classifier}/', exist_ok=True)
        save_model(save_dir=save_dir, classifier=classifier, feature_name=feature_name, fold=fold, model=model)
        save_prediction(save_dir=save_dir, classifier=classifier, feature_name=feature_name, fold=fold, pred=y_pred)

    print("KFOLD acccuracy: ", accuracy_score(y, oof_pred))


if __name__ == '__main__':
    cli()


