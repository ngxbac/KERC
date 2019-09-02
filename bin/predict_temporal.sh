#!/usr/bin/env bash


LOG_DIR=/media/ngxbac/DATA/logs_kerc/lstm/
POOL_FEATURE_DIR=/media/ngxbac/DATA/logs_kerc/features/

for model in inception_v3 inception_v4 se_resnet50 se_resnext50_32x4d; do
    for hidden_size in $(seq 32 32 512); do
        MODEL_LOG_DIR=$LOG_DIR/${model}_${hidden_size}
        python src/infer_temporal.py infer  --log_dir=$MODEL_LOG_DIR \
                                            --model_name=$model \
                                            --hidden_size=$hidden_size \
                                            --feature_pkl=$POOL_FEATURE_DIR/$model/valid/pooled_features.pkl \
                                            --out_dir=$MODEL_LOG_DIR/predict/valid/
    done
done
