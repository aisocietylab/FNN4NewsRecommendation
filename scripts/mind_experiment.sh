#!/usr/bin/env bash

set -e
export TQDM_DISABLE=1

echo "Running MIND"
uv run main.py --config_path=configs/config_mind.json \
    --noskip_validation
uv run eval.py --results_path=results/mind --eval_data_path=datasets/mind_small

echo "Running MIND with last layer set to OR"
uv run main.py --config_path=configs/config_mind.json \
    --results_path=results/mind-lor \
    --layer_logics=and_start \
    --noskip_validation
uv run eval.py --results_path=results/mind-lor --eval_data_path=datasets/mind_small