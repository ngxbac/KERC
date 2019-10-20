#!/usr/bin/env bash
RUN_CONFIG=config.yml

export LC_ALL=C.UTF-8
export LANG=C.UTF-8



#for model in densenet121 fishnet99 fishnet99 inceptionresnetv2 inception_v3 inception_v4 resnet18 resnet34 resnet50 se_resnet50 se_resnext50_32x4d vggresnet; do
#    python src/train_ml.py train-svm    --feature_dir=/media/ngxbac/DATA/logs_kerc/features/$model/ \
#                                        --train_csv=./preprocessing/csv/train_face_clean.csv \
#                                        --valid_csv=./preprocessing/csv/valid_face_clean.csv \
#                                        --config_dir=./ml_configs/ \
#                                        --classifier=svm \
#                                        --feature_name=$model \
#                                        --save_dir=./ml_models/
#done


for model in densenet121 fishnet99 inceptionresnetv2 inception_v3 inception_v4 resnet18 resnet34 resnet50 se_resnet50 se_resnext50_32x4d vggresnet; do
    python src/train_ml.py train-svm-kfold    --feature_dir=/media/ngxbac/DATA/logs_kerc/features/$model/ \
                                        --train_csv=./preprocessing/csv/train_face_clean.csv \
                                        --valid_csv=./preprocessing/csv/valid_face_clean.csv \
                                        --config_dir=./ml_configs/ \
                                        --classifier=svm \
                                        --feature_name=$model \
                                        --save_dir=./ml_models_kfold/
done