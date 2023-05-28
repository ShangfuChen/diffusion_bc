#! /usr/bin/env bash

for ((seed=1; seed<=10; seed++)) do
    echo $seed

    python goal_prox/main.py \
        --alg ibc \
        --prefix record-ibc \
        --env-name maze2d-medium-v2 \
        --traj-load-path "./ibc/maze2d_100.pt" \
        --load-file ./data/trained_models/CustomHandManipulateBlockRotateZ-v2/522-CHMBRZ-3-QD-ibc/model_390000.pt \
        --normalize-env False  \
        --eval-num-processes 1 \
        --num-eval 1 \
        --clip-actions True \
        --bc-state-norm False \
        --il-in-action-norm False \
        --il-out-action-norm False \
        --stochastic-optimizer-train_samples 256\
        --seed ${seed} \
        --num-render 50 \
        --vid-fps 10 \
        --eval-only \
        --hidden-dim 1024 \
        --depth 2 \
        --vid-dir final
  
done
