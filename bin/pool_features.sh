#!/usr/bin/env bash


for model in densenet121 fishnet99 inceptionresnetv2 inception_v3 inception_v4 resnet18 resnet34 resnet50 se_resnet50 se_resnext50_32x4d vggresnet; do
    for dataset in train valid; do
        feature_path=/media/ngxbac/DATA/logs_kerc/features/$model/$dataset/
        csv_file=./preprocessing/csv/${dataset}_face_clean.csv
        python src/pool_features.py pool-features   --features_path=$feature_path \
                                                    --csv_file=$csv_file
    done
done