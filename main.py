#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import random
import torch
import logging

import simple_parsing
import numpy as np
import seaborn as sns

from src import Config, evaluation
from src.data import (
    create_ebnerd_graph_datasets,
    create_mind_graph_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("main.log"),
    ],
)


def parse_args(argv: list[str] | None = None) -> Config:
    return simple_parsing.parse(
        config_class=Config, args=argv, add_config_path_arg=True
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    # set seaborn config for any plots (e.g., loss curves)
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")

    datasets_cache_path = config.data.cache_path
    if datasets_cache_path and os.path.exists(datasets_cache_path):
        with open(datasets_cache_path, "rb") as handle:
            (train_data, val_data, test_data) = pickle.load(handle)
    else:
        if config.data.dataset_type == "mind":
            train_data, val_data, test_data = create_mind_graph_datasets(
                config.data, seed=config.seed, save_path_boolenizers=config.results_path
            )
        else:
            train_data, val_data, test_data = create_ebnerd_graph_datasets(
                config.data, seed=config.seed, save_path_boolenizers=config.results_path
            )

        if datasets_cache_path:
            dirname = os.path.dirname(datasets_cache_path)
            if dirname != "":
                os.makedirs(dirname, exist_ok=True)
            with open(datasets_cache_path, "wb") as handle:
                pickle.dump(
                    (train_data, val_data, test_data),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    logger = logging.getLogger(__name__)

    logger.info(f"Configuration: {config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(config.results_path, exist_ok=True)
    with open(os.path.join(config.results_path, "used_config.json"), "w") as f:
        config.dump_json(f)
    evaluation(
        config,
        config.evaluation.num_repeat_experiment,
        device,
        train_data,
        val_data,
        config.evaluation.used_baselines,
    )


if __name__ == "__main__":
    main()
