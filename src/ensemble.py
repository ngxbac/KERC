import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def load_pred(save_dir, classifier, feature_name):
    return np.load(f'{save_dir}/{feature_name}/{classifier}/valid.npy')


def get_labels(valid_csv):
    valid_df = pd.read_csv(valid_csv)

    le = LabelEncoder()
    valid_df['emotion'] = le.fit_transform(valid_df['emotion'])

    y_valid = valid_df['emotion'].values

    labels = []

    videos = valid_df['video_name'].unique()
    for video in videos:
        video_df = valid_df[valid_df['video_name'] == video]
        y_video = y_valid[video_df.index]
        labels.append(y_video[0])

    return np.asarray(labels)


from ortools.graph import pywrapgraph
def mcf_cal(X, dict_dist):
    # X = X / X.sum(axis=0)
    m = X * 1000000000
    m = m.astype(np.int64)
    nb_rows, nb_classes = X.shape[0], X.shape[1]
    mcf = pywrapgraph.SimpleMinCostFlow()
    # Suppliers: distribution
    for j in range(nb_classes):
        mcf.SetNodeSupply(j + nb_rows, int(dict_dist[j]))
    # Rows
    for i in range(nb_rows):
        mcf.SetNodeSupply(i, -1)
        for j in range(nb_classes):
            mcf.AddArcWithCapacityAndUnitCost(j + nb_rows, i, 1, int(-m[i][j]))
    mcf.SolveMaxFlowWithMinCost()

    assignment = np.zeros(nb_rows, dtype=np.int32)
    for i in range(mcf.NumArcs()):
        if mcf.Flow(i) > 0:
            assignment[mcf.Head(i)] = mcf.Tail(i) - nb_rows
    return assignment


classifier = 'svm'
model_dir = "./ml_models/"
train_csv = './preprocessing/csv/train_face_clean.csv'
valid_csv = './preprocessing/csv/valid_face_clean.csv'
preds = []
y_video_valid = get_labels(valid_csv)
dist_dict = {}
labels, counts = np.unique(y_video_valid, return_counts=True)
for l, c in zip(labels, counts):
    dist_dict[l] = c

for feature_name in [
    "densenet121",
    "fishnet99",
    "inceptionresnetv2",
    # "inception_v3",
    # "inception_v4",
    # "resnet18",
    # "resnet34",
    "resnet50",
    # "se_resnet50",
    "se_resnext50_32x4d",
    # "vggresnet"
]:
    pred = load_pred(model_dir, classifier, feature_name)
    acc = accuracy_score(y_video_valid, pred.argmax(axis=1))
    print("Backbone {},  \t\tacc: {}".format(feature_name, acc))
    preds.append(pred)

preds = np.asarray(preds)
preds = preds.mean(axis=0)
# y_pred = mcf_cal(preds, dist_dict)
y_pred = np.argmax(preds, axis=1)
print(accuracy_score(y_video_valid, y_pred))
