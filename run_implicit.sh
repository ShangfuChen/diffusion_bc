#!/usr/bin/env bash

set -e
set -x

EXPERIMENT_NAME=implicit_ebm_${train_size}

python train_ibc.py \
    --experiment-name $EXPERIMENT_NAME \
    --policy-type IMPLICIT \
    --dropout-prob 0.0 \
    --weight-decay 0.0 \
    --max-epochs 2000 \
    --train_batch_size 128 \
    --lr 1e-3 \
    --spatial-reduction SPATIAL_SOFTMAX \
    --stochastic-optimizer-train-samples 128 \
