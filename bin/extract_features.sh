#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


root=/media/ngxbac/Bac/dataset/KERC/faces/

for model in densenet121 fishnet99 inceptionresnetv2 inception_v3 inception_v4 resnet18 resnet34 resnet50 se_resnet50 se_resnext50_32x4d vggresnet; do
    checkpoint=/media/ngxbac/DATA/logs_emotiw_temporal/feature_extractor/$model/checkpoints/best.pth
    for dataset in train valid; do
        csv_file=./preprocessing/csv/${dataset}_face.csv
        save_feature_dir=/media/ngxbac/DATA/logs_kerc/features/${model}/${dataset}
        python src/extract_features.py extract  --csv_file=$csv_file \
                                                --root=$root \
                                                --model=$model \
                                                --checkpoint=$checkpoint \
                                                --save_feature_dir=$save_feature_dir
    done
done