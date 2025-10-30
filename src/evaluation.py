from collections import defaultdict
from typing import Callable

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import torch
import os
from torch_geometric.data import HeteroData
from surprise import Dataset, Reader
from surprise import (
    NormalPredictor,
    KNNBasic,
    SVDpp,
)
import logging

from src.model import (
    UserItemAtomEmbedding,
    LogicNetwork,
)
from src.utils import (
    Config,
    plot_and_save_loss,
    ALL_BASELINE_NAMES,
)
from src.metrics import evaluation_ranked_metrics


def test_approach(
    config: Config,
    n_repeat_exp,
    device: torch.device,
    train_data: HeteroData,
    evaluation_data: HeteroData,
    learning_rate,
    num_epochs,
    batch_size,
    loss_fn: str | Callable,
    l1_lambda,
):
    edge_name = config.data.rating_edge_name
    predefined_names = train_data[edge_name].edge_label_predefined_names
    num_of_predefined_features = len(predefined_names)

    if isinstance(loss_fn, str):
        loss_fn = getattr(torch.nn, loss_fn)(reduction="none")

    results_at_k = defaultdict(list)
    for i in range(n_repeat_exp):
        # move data to device
        user_ids, item_ids = train_data[edge_name].edge_label_index

        model = LogicNetwork(
            config=config,
            predefined_input_size=num_of_predefined_features,
            layers=config.model.layers,
            embedding_atoms_config=UserItemAtomEmbedding(
                user_ids=user_ids,
                item_ids=item_ids,
                num_user_embeddings=config.model.num_user_embedding_atoms,
                num_item_embeddings=config.model.num_item_embedding_atoms,
            ),
        ).to(device)

        train_losses, val_losses = model.fit(
            train_graph=train_data,
            evaluation_graph=evaluation_data,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            l1_lambda=l1_lambda,
            num_epochs=num_epochs,
            num_negative_samples_per_positive=config.training.num_negative_samples_per_positive,
            optimizer_name=config.training.optimizer,
            batch_size=batch_size,
            device=device,
            epoch_checkpoint_prefix=f"model_run_{i}_epoch_",
        )
        # store the model
        model.eval()
        torch.save(model, os.path.join(config.results_path, f"model_run_{i}.pth"))
        plot_and_save_loss(
            train_losses,
            val_losses,
            os.path.join(config.results_path, f"losses_run_{i}.pdf"),
        )
        ### EVALUATION
        with torch.no_grad():
            predictions_df = model.predict(evaluation_data)
            results_at_k_this_run = evaluation_ranked_metrics(
                impression_ids=predictions_df.index,
                predictions=predictions_df["scores"].to_list(),
                labels=predictions_df["labels"].to_list(),
            )

            logging.info(f"{results_at_k_this_run=}")
            for key in results_at_k_this_run:
                results_at_k[key].append(results_at_k_this_run[key])

    return results_at_k


def test_baseline_decision_tree(
    config: Config,
    n_repeat_exp,
    train_data: HeteroData,
    evaluation_data: HeteroData,
):
    rating_edge_name = config.data.rating_edge_name

    X_train = train_data[rating_edge_name]["edge_label_predefined"]
    X_train = pd.DataFrame(
        X_train.detach().cpu().numpy(),
        columns=train_data[rating_edge_name].edge_label_predefined_names,
    )

    y_train = (
        train_data[config.data.rating_edge_name].edge_label
        >= config.data.good_rating_threshold
    ).float()

    results_at_k = dict()
    for i in range(n_repeat_exp):
        decision_tree_baseline = DecisionTreeClassifier()

        decision_tree_baseline.fit(X_train, y_train)

        def model(graph_data: HeteroData):
            X_evaluation = graph_data[rating_edge_name]["edge_label_predefined"]
            # fine-grained predictions between 0 and 1
            return decision_tree_baseline.predict_proba(X_evaluation)[:, 1]

        # compute scores for all user-item pairs to get top-k results
        results_at_k_this_run = evaluation_ranked_metrics(
            impression_ids=evaluation_data[
                config.data.rating_edge_name
            ].edge_label_global_id,
            predictions=model(evaluation_data),
            labels=evaluation_data[config.data.rating_edge_name].edge_label,
        )
        logging.info(f"{results_at_k_this_run=}")
        if i == 0:
            for key in results_at_k_this_run:
                results_at_k[key] = [results_at_k_this_run[key]]
        else:
            for key in results_at_k_this_run:
                results_at_k[key].append(results_at_k_this_run[key])

    return {key: results_at_k[key] for key in results_at_k}


def get_surprise_trainset(config: Config, graph_data: HeteroData):
    edge_name = config.data.rating_edge_name
    # see also https://surprise.readthedocs.io/en/stable/getting_started.html#use-a-custom-dataset
    ratings = graph_data[edge_name].edge_label
    user_ids = graph_data[edge_name[0]].global_id[
        graph_data[edge_name].edge_label_index[0]
    ]
    item_ids = graph_data[edge_name[-1]].global_id[
        graph_data[edge_name].edge_label_index[1]
    ]
    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {
        "userID": user_ids,
        "itemID": item_ids,
        "rating": ratings,
    }
    df = pd.DataFrame(ratings_dict)
    # A reader is still needed but only the rating_scale param is required.
    reader = Reader(rating_scale=(0, 1))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
    return data.build_full_trainset()


def get_surprise_testset(user_ids: torch.Tensor, item_ids: torch.Tensor, ratings=None):
    # only depend on edge_index, do not need graph
    if ratings is None:
        ratings = -torch.ones_like(user_ids)  # do not know correct ratings for testset

    ratings_dict = {
        "userID": user_ids,
        "itemID": item_ids,
        "rating": ratings,  # do not know correct ratings for testset
    }
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
    return data.construct_testset(data.raw_ratings)


def test_baseline_surprise(
    config: Config, n_repeat_exp, train_data: HeteroData, evaluation_data: HeteroData
):
    trainset = get_surprise_trainset(config, train_data)
    edge_name = config.data.rating_edge_name
    # shuffle evaluation dataset (to get random order for unknown user/items, because our dataset is sorted with positive examples first)
    edge_index = evaluation_data[edge_name].edge_label_index
    evaluation_labels = evaluation_data[edge_name].edge_label
    shuffled_indices = torch.randperm(edge_index.size(1))
    edge_index = edge_index[:, shuffled_indices]
    evaluation_labels = evaluation_labels[shuffled_indices]
    testset = get_surprise_testset(
        # map to actual user and item ids
        evaluation_data[edge_name[0]].global_id[edge_index[0]],
        evaluation_data[edge_name[-1]].global_id[edge_index[1]],
        evaluation_labels,
    )

    results_at_k = dict()
    algorithms = {
        "NormalPredictor": NormalPredictor(),
        "KNNBasic": KNNBasic(),
        "KNNItem": KNNBasic(sim_options={"user_based": False}),
        "SVD++": SVDpp(),
    }

    for name, algo in algorithms.items():
        logging.info(f"Test {name} baseline...")
        if name not in results_at_k:
            results_at_k[name] = {}
        num_runs = n_repeat_exp
        if name in ["BaselineOnly"]:
            num_runs = 1  # deterministic algorithms
        for i in range(num_runs):
            # training
            algo.fit(trainset)

            # evaluation
            predictions = np.array(algo.test(testset))
            scores = predictions[:, 3].astype(float)

            predictions_df = pd.DataFrame(
                predictions, columns=["uid", "iid", "rui", "est", "details"]
            )
            predictions_df.to_csv(
                os.path.join(config.results_path, f"{name}_predictions_run_{i}.csv"),
                index=False,
            )

            # compute scores for all user-item pairs to get top-k results
            results_at_k_this_run = evaluation_ranked_metrics(
                impression_ids=evaluation_data[
                    config.data.rating_edge_name
                ].edge_label_global_id,
                predictions=scores,
                labels=evaluation_data[config.data.rating_edge_name].edge_label,
            )
            logging.info(f"{results_at_k_this_run=}")
            for key in results_at_k_this_run:
                if key not in results_at_k[name]:
                    results_at_k[name][key] = []
                results_at_k[name][key].append(results_at_k_this_run[key])

    return {
        algo: {key: results_at_k[algo][key] for key in results_at_k[algo]}
        for algo in results_at_k
    }


def evaluation(
    config: Config,
    n_repeat_exp: int,
    device: torch.device,
    train_data: HeteroData,
    val_data: HeteroData,
    baselines_to_test: list[str] = None,
):
    learning_rate = config.training.learning_rate
    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    loss_fn = config.training.loss_fn
    l1_lambda = config.training.l1_lambda

    if config.evaluation.baselines_only:
        results_dict = {}
    else:
        approach_results = test_approach(
            config,
            n_repeat_exp,
            device,
            train_data,
            val_data,
            learning_rate,
            num_epochs,
            batch_size,
            loss_fn,
            l1_lambda,
        )
        results_dict = {"Our Model": approach_results}

    baselines = {}

    if baselines_to_test is None:
        baselines_to_test = list(ALL_BASELINE_NAMES)

    for baseline_name in baselines_to_test:
        if baseline_name == "Decision Tree Pred":
            result = test_baseline_decision_tree(
                config, n_repeat_exp, train_data, val_data
            )
        elif baseline_name == "Surprise":
            result = test_baseline_surprise(config, n_repeat_exp, train_data, val_data)
        elif baseline_name in ALL_BASELINE_NAMES:
            raise NotImplementedError(
                f"Baseline {baseline_name} is not implemented yet. Please implement it or remove it from the config."
            )
        else:
            logging.error(f"Unknown baseline {baseline_name}, skipping...")
            continue

        baselines[baseline_name] = result

    for baseline_name, baseline_results in baselines.items():
        if baseline_name not in baselines_to_test:
            logging.info(f"Skipping baseline {baseline_name}")
            continue
        if baseline_name == "Surprise":
            results_dict.update(baseline_results)
        else:
            results_dict[baseline_name] = baseline_results

    df_data = []
    for baseline_name, results in results_dict.items():
        row_values = {"Method": baseline_name}
        for metric, values in results.items():
            row_values[metric] = values

        df_data.append(pd.DataFrame(row_values))
    results_df = pd.concat(df_data, ignore_index=True)
    results_df.to_csv(os.path.join(config.results_path, "results.csv"), index=False)
