#!/usr/bin/env bash

set -e
export TQDM_DISABLE=1

if [ -d datasets/cache ]; then
    echo "Clearing cache"
    rm -r datasets/cache
    mkdir -p datasets/cache
fi

bash scripts/ebnerd_experiment.sh

bash scripts/mind_experiment.sh

echo "Running baselines"
uv run scripts/baselines.sh

echo "Running ablation study"
uv run scripts/ablation_study.sh
uv run scripts/ablation_study.sh --dataset mind

echo "Running sensitivity study"
uv run scripts/sensitivity_study.sh
