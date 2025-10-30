#!/usr/bin/env bash

set -e
export TQDM_DISABLE=1

echo "Running EB-NeRD"
uv run main.py --config_path=configs/config_ebnerd.json \
    --noskip_validation
uv run eval.py --results_path=results/ebnerd --eval_data_path=datasets/ebnerd_small

echo "Running EB-NeRD with last layer OR"
uv run main.py --config_path=configs/config_ebnerd.json \
    --results_path=results/ebnerd-lor \
    --layer_logics=and_start \
    --noskip_validation
uv run eval.py --results_path=results/ebnerd-lor --eval_data_path=datasets/ebnerd_small