#!/usr/bin/env bash
RUN_CONFIG=config.yml


LOGDIR=/media/ngxbac/DATA/logs_kerc/finetune/vggresnet/
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --verbose