import os
import click

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils import *


@click.group()
def cli():
    print("Search parameters")


class Objective(object):
    def __init__(self, classifier, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train

        self.X_valid = X_valid
        self.y_valid = y_valid

        self.classifier = classifier

    def get_classifier(self, classifier, trial):
        if classifier == 'svm':
            C = trial.suggest_loguniform('svc_c', 5e-4, 1)
            clf = SVC(C=C, kernel='linear', random_state=2411)
        elif classifier == 'KNeighborsClassifier':
            leaf_size = trial.suggest_int('leaf_size', 30, 400)
            # algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            clf = KNeighborsClassifier(n_neighbors=7, leaf_size=leaf_size)
        elif classifier == 'RandomForestClassifier':
            n_estimators = trial.suggest_int('n_estimators', 50, 400)
            clf = RandomForestClassifier(n_estimators=n_estimators)
        else:
            raise ValueError("Not support classifier")

        return clf

    def __call__(self, trial):
        clf = self.get_classifier(self.classifier, trial)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        accuracy = accuracy_score(self.y_valid, y_pred)
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

    X_video_train, X_video_valid, y_video_train, y_video_valid = get_data(train_csv, valid_csv, feature_dir)

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


