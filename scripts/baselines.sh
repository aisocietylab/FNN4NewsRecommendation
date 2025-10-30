#!/usr/bin/env bash

set -e
export TQDM_DISABLE=1

echo "Our implementation of the baselines"
echo "Running EB-NeRD baselines only"
uv run main.py --config_path=configs/config_ebnerd.json \
    --results_path=results/ebnerd-baselines \
    --baselines_only \
    --used_baselines "Decision Tree Pred" "Surprise"
echo "Running MIND baselines only"
uv run main.py --config_path=configs/config_mind.json \
    --results_path=results/mind-baselines \
    --baselines_only \
    --used_baselines "Decision Tree Pred" "Surprise"


echo "Running EB-NeRD nrms"
uv run scripts/baselines/nrms_ebnerd.py

echo "Running MIND nrms"
uv run scripts/baselines/nrms_mind.py
