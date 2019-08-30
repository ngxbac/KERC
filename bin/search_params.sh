#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


root=/media/ngxbac/Bac/dataset/KERC/faces/
#fishnet99 inceptionresnetv2 inception_v3 inception_v4 resnet18 resnet34 resnet50 se_resnet50 se_resnext50_32x4d vggresnet
for model in densenet121 fishnet99 inceptionresnetv2 inception_v3 inception_v4 resnet18 resnet34 resnet50 se_resnet50 se_resnext50_32x4d vggresnet; do
    save_feature_dir=/media/ngxbac/DATA/logs_kerc/features/${model}/
    python src/optuna_search.py search-svm  --feature_dir=$save_feature_dir \
                                            --train_csv=./preprocessing/csv/train_face_clean.csv \
                                            --valid_csv=./preprocessing/csv/valid_face_clean.csv \
                                            --n_trials=200 \
                                            --feature_name=$model \
                                            --out_config=./ml_configs/ \
                                            --classifier=svm
done