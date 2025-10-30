#!/usr/bin/env bash

set -e
export TQDM_DISABLE=1


USAGE="Usage: $0 [--dataset [ebnerd|mind]]"
INCLUDE_DEPRECATED=0
DATASET="ebnerd"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        *) echo $USAGE; exit 1 ;;
    esac
    shift
done

if [ "$DATASET" == "ebnerd" ]; then
    config_file=configs/config_ebnerd-ablation.json
elif [ "$DATASET" == "mind" ]; then
    config_file=configs/config_mind-ablation.json
else
    echo "Unknown dataset: $DATASET"
    echo $USAGE
    exit 1
fi

echo "Running base"
uv run main.py --config_path=$config_file \
    --results_path=results/$DATASET-ablat_base

echo "Running with last layer set to OR"
uv run main.py --config_path=$config_file \
    --results_path=results/$DATASET-ablat_lastlayer-or \
    --layer_logics=and_start

echo "Running without negative sampling (use all negatives)"
uv run main.py --config_path=$config_file \
    --results_path=results/$DATASET-ablat_no-negative-sampling \
    --num_negative_samples_per_positive \
    --noperform_repeated_sampling

echo "Running with MSELoss"
uv run main.py --config_path=$config_file \
    --results_path=results/$DATASET-ablat_mse-loss \
    --loss_fn=MSELoss

echo "Running without l1 loss"
uv run main.py --config_path=$config_file \
    --results_path=results/$DATASET-ablat_no-l1 \
    --l1_lambda=0

echo "Running without orthogonal loss"
uv run main.py --config_path=$config_file \
    --results_path=results/$DATASET-ablat_no-orthogonal \
    --orthogonal_lambda=0

# Experiment with changes to dataset -> set cache_path to None
echo "Running without article age"
uv run main.py --config_path=$config_file \
    --results_path=results/$DATASET-ablat_no-article-age \
    --noadd_article_age_atoms \
    --cache_path

echo "Running with Sigmoid"
uv run main.py --config_path=$config_file \
    --results_path=results/$DATASET-ablat_sigmoid \
    --activation_function=Sigmoid