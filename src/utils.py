import os
import pickle
from pathlib import Path
from typing import Literal, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from simple_parsing.helpers import Serializable
import seaborn as sns


### config ###


@dataclass
class DatasetConfig:
    name: str
    good_rating_threshold: float
    rating_edge_name: Tuple
    base_path: str
    dataset_type: Literal["synthetic", "ebnerd", "mind"] = "ebnerd"

    test_path: str | None = None
    cache_path: str | None = None

    fraction: float | None = None

    # Atom configuration
    num_keep_frequent_classes: int = 20
    add_article_age_atoms: bool = True
    impute_strategy: Literal["mean", "median", "true", "0.5", "false"] = "mean"


@dataclass
class ModelConfig:
    activation_function: Literal["Sigmoid", "Hardsigmoid", "Tanh"] = "Tanh"
    """Activation function to use in the model."""
    initialization: Literal["xavier", "kaiming", "orthogonal", "lecun"] = "kaiming"
    orthogonal_loss_method: Literal["Orth", "SoftOrth", "DoubleSoftOrth"] = (
        "DoubleSoftOrth"
    )

    layers: list[int] = field(default_factory=lambda: [48, 32, 16, 1])
    concat_negated_atoms: bool = False
    """Whether to concatenate negated atoms to the input of the model"""
    with_negated_node_outputs: bool = False
    """Whether to add an output for each node, negating the actual output value"""

    dropout: float | None = None
    num_user_embedding_atoms: int = 0
    num_item_embedding_atoms: int = 0
    boolenize_embeddings: bool = False

    layer_logics: Literal["and_start", "or_start", "and_only", "or_only"] = "or_start"
    """Logic operations to use in the layers of the model. Options are:
    - `and_start`: start with AND logic and alternate with OR logic.
    - `or_start`: start with OR logic and alternate with AND logic.
    - `and_only`: use only AND logic.
    - `or_only`: use only OR logic."""
    last_layer_logic: Literal["and", "or", None] = None
    """Force the last layer to use a specific logic operation.
    If None, the logic is given by `layer_logics`."""


@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    num_epochs: int = 20
    batch_size: int = 1024 * 8
    optimizer: str = "AdamW"
    loss_fn: str = "BCELoss"
    l1_lambda: float = 0.1
    orthogonal_lambda: float = 0.1

    num_negative_samples_per_positive: int | None = 4

    save_model_every_n_epochs: int | None = None
    skip_validation: bool = False


ALL_BASELINE_NAMES = [
    "Decision Tree Pred",
    "Surprise",
]


@dataclass
class EvaluationConfig:
    num_repeat_experiment: int
    used_baselines: list[str] = field(default_factory=lambda: ALL_BASELINE_NAMES.copy())
    baselines_only: bool = False


@dataclass
class Config(Serializable):
    data: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    results_path: str
    seed: int | None = 42


### eval/prediction ###


def group_predictions_by_impression(
    impression_id: np.ndarray, scores: np.ndarray, labels: np.ndarray | None = None
) -> pd.DataFrame:
    assert len(impression_id) == len(scores), f"{len(impression_id)} != {len(scores)}"
    if labels is None:
        data = {"scores": scores}
    else:
        assert len(impression_id) == len(labels), (
            f"{len(impression_id)} != {len(labels)}"
        )
        data = {"scores": scores, "labels": labels}
    prediction_df = pd.DataFrame(index=impression_id, data=data)

    return prediction_df.groupby(by=prediction_df.index).agg(list)


### visualisation ###


def plot_and_save_loss(train_losses, val_losses, filename):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    train_data = pd.DataFrame(
        {
            "Loss": train_losses,
            "Epoch": range(1, len(train_losses) + 1),
            "Type": "Train",
        }
    )
    eval_data = pd.DataFrame(
        {
            "Loss": val_losses,
            "Epoch": range(1, len(val_losses) + 1),
            "Type": "Validation",
        }
    )
    data = pd.concat((train_data, eval_data))

    figure = sns.relplot(data=data, kind="line", x="Epoch", y="Loss", hue="Type")
    figure.set_titles("Training Loss")

    figure.savefig(filename, format="pdf")
    plt.close()


### Miscellaneous ###


def load_boolenizers_and_imputer(results_path: str | os.PathLike) -> Tuple:
    results_path = Path(results_path)
    with open(results_path / "article_boolenizer.pkl", "rb") as f:
        articles_boolenizer = pickle.load(f)
    with open(results_path / "behavior_boolenizer.pkl", "rb") as f:
        behaviors_boolenizer = pickle.load(f)
    with open(results_path / "imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    return articles_boolenizer, behaviors_boolenizer, imputer


def create_optimizer(optimizer_name: str, param_group) -> torch.optim.Optimizer:
    weight_decay = 1e-5

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(param_group, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(param_group, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            param_group, weight_decay=weight_decay, momentum=0.9, nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer


def rule_complexity(expr):
    if len(expr.args) == 0:
        return 1
    return sum(rule_complexity(arg) for arg in expr.args)
