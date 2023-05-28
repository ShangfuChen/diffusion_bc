#! /usr/bin/env bash

for ((seed=1; seed<=10; seed++)) do
    echo $seed

    python goal_prox/main.py \
        --alg diff-policy \
        --prefix record-dp \
        --env-name CustomHandManipulateBlockRotateZ-v2 \
        --load-file ./models/hand_dp \
        --traj-load-path "./expert_datasets/hand_10000_v2.pt" \
        --normalize-env False  \
        --eval-num-processes 1 \
        --num-eval 1 \
        --clip-actions True \
        --bc-state-norm True \
        --seed ${seed} \
        --num-render 50 \
        --vid-fps 10 \
        --eval-only \
        --hidden-dim 2100 \
        --depth 4 \
        --vid-dir final
  
done