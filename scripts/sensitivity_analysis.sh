#!/usr/bin/env bash

set -e
export TQDM_DISABLE=1

echo "Sensitivity study for l1 lambda"
for l1_lambda in 0.001 0.00316228 0.01 0.03162278 0.1 0.31622777 1.0; do
    echo "Running with l1 lambda: $l1_lambda"
    uv run main.py --config_path=configs/config_ebnerd-sensitivity.json \
        --results_path="results/ebnerd-sens_l1lambda_$l1_lambda" \
        --l1_lambda=$l1_lambda
done

echo "Sensitivity study for orthogonal lambda"
for ortho_lambda in 0.001 0.00316228 0.01 0.03162278 0.1 0.31622777 1.0 3.1622778 10.0; do
    echo "Running with orthogonal lambda: $ortho_lambda"
    uv run main.py --config_path=configs/config_ebnerd-sensitivity.json \
        --results_path="results/ebnerd-sens_ortholambda_$ortho_lambda" \
        --orthogonal_lambda=$ortho_lambda
done

echo "Sensitivity study for batch size"
for batch_size in 256 512 1024 2048 4096 8192 16384 32768 65536; do
    echo "Running with batch size: $batch_size"
    uv run main.py --config_path=configs/config_ebnerd-sensitivity.json \
        --results_path="results/ebnerd-sens_batchsize_$batch_size" \
        --batch_size=$batch_size
done