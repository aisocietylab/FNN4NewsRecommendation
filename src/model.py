from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math
import os
from typing import Iterable, Literal

import sympy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import HeteroData


from src.utils import (
    Config,
    create_optimizer,
    group_predictions_by_impression,
    rule_complexity,
)


torch.set_printoptions(sci_mode=False)


@dataclass
class FuzzyOperators(ABC):
    logspace: bool = False

    @abstractmethod
    def lnot(self, x):
        raise NotImplementedError

    @abstractmethod
    def land(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def lor(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def lor_aggregate(self, variables):
        raise NotImplementedError

    @abstractmethod
    def land_aggregate(self, variables):
        raise NotImplementedError


@dataclass
class ProductTNormOperators(FuzzyOperators):
    def lnot(self, x: torch.Tensor):
        return 1 - x

    def land(self, x: torch.Tensor, y: torch.Tensor):
        return x * y

    def lor(self, x: torch.Tensor, y: torch.Tensor):
        return x + y - x * y

    def land_aggregate(self, variables: torch.Tensor, reduction_dim=-1):
        """
        Apply product t norm and on multiple variables
        """
        return torch.prod(variables, dim=reduction_dim)

    def lor_aggregate(self, variables: torch.Tensor | np.ndarray, reduction_dim=-1):
        """
        Apply OR recursively, to create OR with more than 2 inputs
        """
        # 1 - `the probability that all variables are 0`/`all events did not happen`
        return self.lnot(self.land_aggregate(self.lnot(variables), reduction_dim))


class LogicLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        fuzzy_operators: FuzzyOperators = None,
        with_negated_outputs: bool = False,
        activation_function: str = "Sigmoid",
        logic_operation: Literal["and", "or"] = "and",
        initialization: Literal["normal", "uniform", "xavier", "orthogonal"] = "normal",
        dropout: float | None = None,
    ):
        super().__init__()

        if fuzzy_operators is None:
            fuzzy_operators = ProductTNormOperators()

        if with_negated_outputs:
            assert out_features % 2 == 0, (
                f"Expected {out_features=} to be even, because half are used for negated outputs."
            )

        num_output_weights = out_features // 2 if with_negated_outputs else out_features

        self.ops = fuzzy_operators
        self.weights = nn.Parameter(torch.empty((num_output_weights, in_features)))
        self.with_negated_outputs = with_negated_outputs
        self.activation_function = getattr(nn, activation_function)()
        self.logic_operation = logic_operation
        self.dropout = dropout

        # initialise weights (gain=1, because we use sigmoid)
        initialization = initialization.lower()
        # TODO: map some activation functions to `sigmoid`
        nonlinearity = activation_function.lower()
        if nonlinearity == "hardsigmoid":
            # Hardsigmoid needs the same gain as sigmoid
            nonlinearity = "sigmoid"

        gain = nn.init.calculate_gain(nonlinearity)
        if initialization == "xavier":
            # recommended for `sigmoid`
            nn.init.xavier_uniform_(self.weights, gain=gain)
        elif initialization == "kaiming":
            nn.init.kaiming_uniform_(self.weights, nonlinearity=nonlinearity)
        elif initialization == "orthogonal":
            nn.init.orthogonal_(self.weights, gain=gain)
        elif initialization == "lecun":
            nn.init.uniform_(
                self.weights, a=-1 / np.sqrt(in_features), b=1 / np.sqrt(in_features)
            )
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

    def forward(self, atoms: torch.Tensor, extraction_threshold=None):
        atoms = atoms.float()

        assert atoms.ndim == 2
        assert atoms.shape[1] == self.weights.shape[1], (
            f"Expected shape {(atoms.shape[0], self.weights.shape[1])}, but got {atoms.shape}"
        )
        # Prepare for broadcast for `num_nodes`: (batch, num_nodes, num_atoms)
        atoms = atoms.unsqueeze(1)

        weighted_atoms = self.weight_atoms(atoms, extraction_threshold)

        # FIXME: bias behavior is not good yet!
        if self.logic_operation == "and":
            y = self.ops.land_aggregate(weighted_atoms)
        else:
            # pull value to `0`
            y = self.ops.lor_aggregate(weighted_atoms)

        assert y.shape == (
            atoms.shape[0],
            self.weights.shape[0],
        ), (
            f"Expected shape {(atoms.shape[0], self.weights.shape[0])}, but got {y.shape}"
        )
        if self.with_negated_outputs:
            y = torch.cat((y, self.ops.lnot(y)), dim=-1)

        return y

    def weight_atoms(
        self, atoms: torch.Tensor, extraction_threshold: float | None = None
    ):
        assert atoms.ndim == 3, f"Expected 3D tensor, but got {atoms.shape=}"
        # apply sigmoid to weights to get values between 0 and 1
        fuzzy_weights = self.get_fuzzy_weights(extraction_threshold).to(
            device=atoms.device
        )
        assert fuzzy_weights.ndim == 2

        # negative weights (e.g. tanh activation), should be treated as a weight on the negated atom.
        zero_weights = fuzzy_weights < 0
        if zero_weights.any():
            # This function, though not differentiable at 0, there is no jump in function value.
            atom_negation_mask = zero_weights.unsqueeze(0)

            atoms = torch.where(
                atom_negation_mask,
                self.ops.lnot(atoms),
                atoms,
            )
            # take absolute of weights, negative weights don't make sense
            fuzzy_weights = torch.abs(fuzzy_weights)

        # select inputs based on weights
        # the weight basically pulls the atom towards the neutral element (0 for OR, 1 for AND)
        if self.logic_operation == "and":
            # identical to `_or_` (pulling towards neutral element):
            # y = 1 - (1 - atoms) * fuzzy_weights
            return self.ops.lor(atoms, self.ops.lnot(fuzzy_weights))
        else:
            # pull value to `0`
            return self.ops.land(atoms, fuzzy_weights)

    def get_fuzzy_weights(self, extraction_threshold=None):
        fuzzy_weights = self.activation_function(self.weights)

        if extraction_threshold:
            fuzzy_weights = torch.where(
                torch.abs(fuzzy_weights) < extraction_threshold, 0.0, fuzzy_weights
            )
        return fuzzy_weights


@dataclass
class UserItemAtomEmbedding:
    user_ids: torch.Tensor
    item_ids: torch.Tensor

    num_user_embeddings: int
    num_item_embeddings: int

    @torch.no_grad()
    def __post_init__(self):
        self.user_ids = torch.tensor(np.unique(self.user_ids.detach().cpu().numpy()))
        self.item_ids = torch.tensor(np.unique(self.item_ids.detach().cpu().numpy()))

    def get_total_num_embedding_atoms(self):
        return self.num_user_embeddings + self.num_item_embeddings

    def create_user_embedding(self):
        if self.num_user_embeddings == 0:
            return None
        return nn.Embedding(
            num_embeddings=len(self.user_ids),
            embedding_dim=self.num_user_embeddings,
        )

    def create_item_embedding(self):
        if self.num_item_embeddings == 0:
            return None
        return nn.Embedding(
            num_embeddings=len(self.item_ids),
            embedding_dim=self.num_item_embeddings,
        )

    # TODO: return a `padding_idx` if `global_user_ids` is not in `user_ids`! These are val or test only users! Same for articles
    def get_user_embeddings(
        self, user_embeddings: nn.Embedding, global_user_ids: torch.Tensor
    ) -> torch.Tensor:
        # find the index `idx` (`self.user[idx]`) of elements in `user_ids`
        assert global_user_ids.ndim == 1
        matches_user_ids = global_user_ids.view(-1, 1) == self.user_ids.to(
            global_user_ids.device
        )
        unknown_user_rows = torch.any(matches_user_ids, dim=-1)

        # Adapted from: https://discuss.pytorch.org/t/how-to-map-input-ids-to-a-limited-embedding-indexes/138587/2
        idx = matches_user_ids.int().argmax(dim=-1)
        embeddings = user_embeddings(idx)

        # TODO: consider using different methods for unknown users (e.g., learn/set padding embedding)
        embeddings[unknown_user_rows] = torch.mean(user_embeddings.weight.data, dim=0)

        return embeddings

    def get_item_embeddings(
        self, item_embeddings: nn.Embedding, global_item_ids: torch.Tensor
    ) -> torch.Tensor:
        # find the index `idx` (`self.item[idx]`) of elements in `item_ids`
        assert global_item_ids.ndim == 1
        matches_item_ids = global_item_ids.view(-1, 1) == self.item_ids.to(
            global_item_ids.device
        )
        unknown_item_rows = torch.any(matches_item_ids, dim=-1)

        # Adapted from: https://discuss.pytorch.org/t/how-to-map-input-ids-to-a-limited-embedding-indexes/138587/2
        idx = matches_item_ids.int().argmax(dim=-1)
        embeddings = item_embeddings(idx)

        # TODO: consider using different methods for unknown items (e.g., learn/set padding embedding)
        embeddings[unknown_item_rows] = torch.mean(item_embeddings.weight.data, dim=0)

        return embeddings


@dataclass
class MyDataset(torch.utils.data.Dataset):
    graph_data: HeteroData

    negative_sampling_ratio: int | None = None
    with_labels: bool = False
    rating_edge_name: tuple[str] = ("user", "rates", "item")

    @property
    def all_impression_ids(self):
        return self.graph_data[self.rating_edge_name].edge_label_global_id

    @property
    def all_inputs(self):
        return self.graph_data[self.rating_edge_name].edge_label_predefined

    @property
    def all_labels(self):
        assert self.with_labels
        return self.graph_data[self.rating_edge_name].edge_label

    @property
    def all_edge_idx(self):
        return self.graph_data[self.rating_edge_name].edge_label_index

    def __post_init__(self):
        edge_idx = self.all_edge_idx
        from_name, _relation_name, to_name = self.rating_edge_name
        self.all_user_ids = self.graph_data[from_name].global_id[edge_idx[0]]
        self.all_item_ids = self.graph_data[to_name].global_id[edge_idx[1]]

        if self.negative_sampling_ratio is None:
            return
        elif self.negative_sampling_ratio <= 0:
            self.negative_sampling_ratio = None
            return
        assert hasattr(self.graph_data[self.rating_edge_name], "edge_label"), (
            "Graph data must have edge labels for negative sampling"
        )
        self.negative_indices_per_impression = {}
        impression_ids = self.all_impression_ids

        labels = self.all_labels
        self.positive_samples_mask = labels > 0

        self.negative_indices_per_impression = (
            pd.DataFrame(
                {
                    "impression_id": impression_ids.numpy(),
                    # "article_id": article_ids.numpy()[~self.positive_samples_mask],
                    "graph_idx": range(len(impression_ids)),
                }
            )[~self.positive_samples_mask.numpy()]
            .groupby("impression_id")
            .agg(list)
        )

        self.sampled_mask = self.positive_samples_mask.clone()
        for impression_id in self.negative_indices_per_impression.index:
            negative_samples = self.negative_indices_per_impression.loc[
                impression_id, "graph_idx"
            ]
            sampled_negatives = np.random.choice(
                negative_samples,
                size=min(self.negative_sampling_ratio, len(negative_samples)),
                replace=False,
            )

            self.sampled_mask[sampled_negatives] = True

        self.sampled_impression_ids = self.all_impression_ids[self.sampled_mask]
        self.sampled_inputs = self.all_inputs[self.sampled_mask]
        self.sampled_user_ids = self.all_user_ids[self.sampled_mask]
        self.sampled_item_ids = self.all_item_ids[self.sampled_mask]
        self.sampled_labels = self.all_labels[self.sampled_mask]

    def _sample_negative_impressions(
        self, impression_ids: Iterable[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.negative_sampling_ratio is None or self.negative_sampling_ratio < 0:
            raise ValueError(
                f"Negative sampling ratio must be greater equal than 0. Got {self.negative_sampling_ratio=}"
            )
        if self.negative_sampling_ratio == 0:
            return None, None, None

        inputs = []
        user_ids = []
        article_ids = []

        if (
            len(self.negative_indices_per_impression.index.intersection(impression_ids))
            == 0
        ):
            logging.debug(
                f"Some impression_ids do not have negative samples: {set(impression_ids) - set(self.negative_indices_per_impression.index)}"
            )
            return None, None, None
        for graph_indices in self.negative_indices_per_impression.loc[impression_ids][
            "graph_idx"
        ].values:
            graph_indices = np.asarray(graph_indices, dtype=int)
            if len(graph_indices) >= self.negative_sampling_ratio:
                negative_sample_idx = np.random.choice(
                    graph_indices, size=self.negative_sampling_ratio, replace=False
                )
            else:
                negative_sample_idx = np.random.choice(
                    graph_indices, size=self.negative_sampling_ratio, replace=True
                )
            inputs.append(self.all_inputs[negative_sample_idx])
            user_ids.append(self.all_user_ids[negative_sample_idx])
            article_ids.append(self.all_item_ids[negative_sample_idx])
        # TODO: support inputs in sparse format
        return (
            np.concatenate(inputs),
            np.concatenate(user_ids),
            np.concatenate(article_ids),
        )

    def __len__(self):
        if self.negative_sampling_ratio is None or self.negative_sampling_ratio <= 0:
            return self.all_impression_ids.shape[0]
        else:
            return self.sampled_mask.sum().item()

    def __getitem__(self, idx: int):
        input = self.all_inputs
        global_user_id = self.all_user_ids
        global_article_id = self.all_item_ids
        if self.with_labels:
            label = self.all_labels

        if self.negative_sampling_ratio is None:
            if self.with_labels:
                return (
                    input[idx],
                    global_user_id[idx],
                    global_article_id[idx],
                    label[idx],
                )
            else:
                return input[idx], global_user_id[idx], global_article_id[idx]
        else:
            input = self.sampled_inputs[idx]
            global_user_id = self.sampled_user_ids[idx]
            global_article_id = self.sampled_item_ids[idx]
            if self.with_labels:
                label = self.sampled_labels[idx]
                return input, global_user_id, global_article_id, label
            else:
                return input, global_user_id, global_article_id


class LogicNetwork(nn.Module):
    def __init__(
        self,
        layers: list[int],
        predefined_input_size: int,
        config: Config,
        fuzzy_operators: ProductTNormOperators | None = None,
        embedding_atoms_config: UserItemAtomEmbedding | None = None,
    ):
        # TODO: most arguments are included in the `Config`. We could use the config instead of passing them individually.
        super().__init__()

        if fuzzy_operators is None:
            fuzzy_operators = ProductTNormOperators()

        self.ops = fuzzy_operators
        self.config = config
        self.predefined_input_size = predefined_input_size
        self.layer_sizes = layers

        assert len(self.layer_sizes) > 0, "At least one layer is required"
        assert self.layer_sizes[-1] == 1, (
            f"The last layer must have exactly one output, which is the final prediction. got: {self.layer_sizes=}"
        )

        self.orthogonal_loss_method = config.model.orthogonal_loss_method
        self.dropout = config.model.dropout
        self.concat_negated_atoms = config.model.concat_negated_atoms
        self.with_negated_node_outputs = config.model.with_negated_node_outputs
        self.embedding_atoms_config = embedding_atoms_config

        assert not self.dropout, (
            "TODO: apply dropout only to input (set atoms=0.5?, or marginalize?)"
        )

        if self.embedding_atoms_config:
            self.item_embedding = embedding_atoms_config.create_item_embedding()
            self.user_embedding = embedding_atoms_config.create_user_embedding()
        else:
            self.item_embedding = None
            self.user_embedding = None
        self.boolenize_embeddings = config.model.boolenize_embeddings

        self.body_size = (
            self.predefined_input_size
            + self.embedding_atoms_config.get_total_num_embedding_atoms()
        )
        if self.concat_negated_atoms:
            # input has double the size
            self.body_size *= 2
        if self.with_negated_node_outputs:
            # each layer has double the size (except output)
            self.layer_sizes = [
                size * 2 for size in self.layer_sizes[:-1]
            ] + self.layer_sizes[-1:]

        if self.config.model.layer_logics == "and_start":
            logic_operations = [
                "and" if i % 2 == 0 else "or" for i in range(len(self.layer_sizes))
            ]
        elif self.config.model.layer_logics == "or_start":
            logic_operations = [
                "and" if i % 2 == 1 else "or" for i in range(len(self.layer_sizes))
            ]
        elif self.config.model.layer_logics == "and_only":
            logic_operations = ["and"] * len(self.layer_sizes)
        elif self.config.model.layer_logics == "or_only":
            logic_operations = ["or"] * len(self.layer_sizes)
        else:
            raise ValueError()

        self.last_layer_logic = self.config.model.last_layer_logic
        if self.last_layer_logic is not None:
            logging.info(
                f"Setting last layer logic to `{self.last_layer_logic}`, instead of `{logic_operations[-1]}`."
            )
            logic_operations[-1] = self.last_layer_logic

        modules = []
        for i, (
            in_features,
            out_features,
            additional_skip_connections,
            logic_operation,
        ) in enumerate(
            zip(
                [self.body_size] + self.layer_sizes[:-1],
                self.layer_sizes,
                [0, self.body_size] + self.layer_sizes[:-2],
                logic_operations,
            )
        ):
            is_last_layer = i >= len(self.layer_sizes) - 1
            if is_last_layer:
                with_negated_outputs = False
            else:
                with_negated_outputs = self.with_negated_node_outputs

            modules.append(
                LogicLayer(
                    in_features,
                    out_features,
                    fuzzy_operators=self.ops,
                    with_negated_outputs=with_negated_outputs,
                    activation_function=self.config.model.activation_function,
                    logic_operation=logic_operation,
                    initialization=self.config.model.initialization,
                )
            )

        if self.config.model.activation_function == "Tanh":
            if self.user_embedding or self.item_embedding:
                logging.warning(
                    "Sigmoid will be used instead of Tanh for embeddings (negative embeddings are invalid!)"
                )
            self.embeddings_activation_function = nn.Sigmoid()
        else:
            self.embeddings_activation_function = getattr(
                nn, self.config.model.activation_function
            )()

        self.layers = nn.ModuleList(modules)

    def get_constructor_arguments(self):
        return {
            "layers": self.layer_sizes,
            "predefined_input_size": self.predefined_input_size,
            "config": self.config,
            "fuzzy_operators": self.ops,
            "embedding_atoms_config": self.embedding_atoms_config,
        }

    def create_dataset_for_graph(
        self, graph_data: HeteroData, with_labels: bool = False
    ):
        return MyDataset(
            graph_data,
            rating_edge_name=self.config.data.rating_edge_name,
            with_labels=with_labels,
        )

    def create_dataloader_for_graph(
        self,
        graph_data: HeteroData | MyDataset,
        batch_size: int = 256,
        with_labels: bool = False,
        pin_memory: bool = False,
    ):
        if isinstance(graph_data, HeteroData):
            graph_data = self.create_dataset_for_graph(
                graph_data, with_labels=with_labels
            )

        dataloader = torch.utils.data.DataLoader(
            graph_data,
            batch_size=batch_size,
            drop_last=False,
            pin_memory=pin_memory,
        )
        return dataloader

    def predict(
        self,
        graph_data: HeteroData,
        with_labels: bool = True,
        group_by_impression: bool = False,
    ):
        with torch.no_grad():
            self.eval()
            predictions = self(graph_data)

        # TODO: group predictions by impression_id and evaluate
        # %%
        prediction_impression_ids = graph_data[
            "user", "rates", "item"
        ].edge_label_global_id

        if group_by_impression:
            return group_predictions_by_impression(
                prediction_impression_ids.cpu().numpy(),
                predictions.cpu().numpy(),
                (
                    graph_data["user", "rates", "item"].edge_label.cpu().numpy()
                    if with_labels
                    else None
                ),
            )
        else:
            prediction_df = pd.DataFrame(
                index=prediction_impression_ids.cpu().numpy(),
                data={"scores": predictions.cpu().numpy()},
            )
            if with_labels:
                labels = graph_data["user", "rates", "item"].edge_label
                prediction_df["labels"] = labels.cpu().numpy()
            return prediction_df

    def forward(
        self,
        graph_data: (
            HeteroData
            | torch.utils.data.DataLoader
            | torch.Tensor
            | torch.utils.data.Dataset
        ),
        user_ids: torch.Tensor | None = None,  # required in case of `torch.Tensor`
        item_ids: torch.Tensor | None = None,
        extraction_threshold=None,
        dataloader_has_labels: bool = False,
    ):
        if isinstance(graph_data, HeteroData) or isinstance(
            graph_data, torch.utils.data.Dataset
        ):
            dataloader = self.create_dataloader_for_graph(
                graph_data, with_labels=dataloader_has_labels
            )
            return self.forward_dataloader(
                dataloader, with_labels=dataloader_has_labels
            )
        elif isinstance(graph_data, torch.utils.data.DataLoader):
            return self.forward_dataloader(
                graph_data, with_labels=dataloader_has_labels
            )
        else:
            assert user_ids is not None and item_ids is not None, (
                "User IDs and Item IDs must be provided when using a tensor."
            )
            return self.forward_tensor(
                graph_data, user_ids, item_ids, extraction_threshold
            )

    def forward_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        with_labels=False,
        progress_bar_desc="Batches",
        **tqdm_kwargs,
    ):
        model_device = self.layers[0].weights.device
        predictions = torch.empty(0, device=model_device)
        # for batch in tqdm(dataloader, desc="Batches", unit="batch"):
        for batch in tqdm(
            dataloader, desc=progress_bar_desc, unit="batch", **tqdm_kwargs
        ):
            if with_labels:
                predefined_atoms, user_ids, item_ids, _labels = batch
            else:
                predefined_atoms, user_ids, item_ids = batch

            predefined_atoms = predefined_atoms.to(device=model_device)
            user_ids = user_ids.to(device=model_device)
            item_ids = item_ids.to(device=model_device)

            y_pred = self.forward_tensor(predefined_atoms, user_ids, item_ids)
            predictions = torch.cat((predictions, y_pred), dim=0)

        return predictions

    def forward_tensor(
        self,
        predefined_atoms: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        extraction_threshold=None,
    ):
        x = predefined_atoms

        # TODO: it is essential that we have "neutral" embedding (pad embedding)
        #   for users and articles not seen during training!
        # concat embeddings. Users first
        if self.user_embedding:
            assert self.embedding_atoms_config is not None
            user_embeddings = self.embedding_atoms_config.get_user_embeddings(
                self.user_embedding, user_ids
            )
            if self.boolenize_embeddings:
                user_embeddings = (user_embeddings > 0.5).float()
            x = torch.cat(
                (
                    x,
                    self.embeddings_activation_function(user_embeddings),
                ),
                dim=-1,
            )
        if self.item_embedding:
            assert self.embedding_atoms_config is not None
            item_embeddings = self.embedding_atoms_config.get_item_embeddings(
                self.item_embedding, item_ids
            )
            if self.boolenize_embeddings:
                item_embeddings = (item_embeddings > 0.5).float()
            x = torch.cat(
                (
                    x,
                    self.embeddings_activation_function(item_embeddings),
                ),
                dim=-1,
            )

        if self.concat_negated_atoms:
            x = torch.cat((x, self.ops.lnot(x)), dim=-1)

        for layer in self.layers:
            x = layer(x, extraction_threshold)

        assert x.ndim == 2 and x.shape[1] == 1, (
            f"Final output shape needs to be (batch_size, 1), but got {x.shape}"
        )
        return x.squeeze(-1)  # remove last dimension, so we have shape (batch_size,)

    def get_fuzzy_weights(
        self, extraction_threshold=None
    ) -> list[list[list[tuple[int, float]]]]:
        # TODO: improve fuzzy weight extraction (refactor)
        all_fuzzy_weights = []

        for i, horn_layer in enumerate(self.layers):
            layer_fuzzy_weights = []

            fuzzy_weights = horn_layer.get_fuzzy_weights(extraction_threshold)

            for node_weights in fuzzy_weights:
                node_fuzzy_weights = []
                for input_idx, weight in enumerate(node_weights):
                    if extraction_threshold is None or weight >= extraction_threshold:
                        node_fuzzy_weights.append((input_idx, weight))

                layer_fuzzy_weights.append(node_fuzzy_weights)
            all_fuzzy_weights.append(layer_fuzzy_weights)

        return all_fuzzy_weights

    def get_fuzzy_weights_flattened(self) -> torch.Tensor:
        return torch.cat([layer.get_fuzzy_weights().flatten() for layer in self.layers])

    def _compute_sparsity_loss(self):
        # FIXME: if dropout is used, only the active weights should be considered by the l1_loss!
        fuzzy_weights_flattened = self.get_fuzzy_weights_flattened()
        l1_loss = torch.sum(torch.abs(fuzzy_weights_flattened))

        return l1_loss

    def _compute_orthogonality_loss(self):
        """Encourages different nodes to use different input atoms."""
        loss = 0
        for layer in self.layers:
            weights = layer.get_fuzzy_weights()
            # each column should correspond to the weights of one output node (so we can use the identical formulas of the papers)
            weights = weights.T  # shape: (in_features, out_features)
            identity = torch.eye(weights.shape[1], device=weights.device)

            if self.orthogonal_loss_method == "Orth":
                loss += torch.sum(
                    torch.abs(torch.matmul(weights.T, weights) - identity)
                )
            elif self.orthogonal_loss_method == "SoftOrth":
                loss += torch.sum(
                    torch.square(torch.matmul(weights.T, weights) - identity)
                )
            elif self.orthogonal_loss_method == "DoubleSoftOrth":
                identity2 = torch.eye(weights.shape[0], device=weights.device)
                loss += torch.sum(
                    torch.square((torch.matmul(weights.T, weights) - identity))
                ) + torch.sum(
                    torch.square(torch.matmul(weights, weights.T) - identity2)
                )
            else:
                raise NotImplementedError(
                    f"Unknown orthogonality method: {self.orthogonal_loss_method}"
                )
        return loss

    def _negative_sampling(
        self,
        negative_dataset: torch.utils.data.Dataset,
        negative_sample_indices_per_user: dict[int, torch.Tensor],
        user_ids_to_sample: Iterable[int] | torch.Tensor,  # can contain duplicates!
        num_negative_samples: int,
    ) -> torch.utils.data.TensorDataset:
        if isinstance(user_ids_to_sample, torch.Tensor):
            user_ids_to_sample = user_ids_to_sample.tolist()
        user_ids_to_sample = np.asarray(user_ids_to_sample)

        negative_samples_indices = []

        for user_id in np.unique(user_ids_to_sample):
            negative_indices = negative_sample_indices_per_user[user_id]

            num_positive_samples = np.count_nonzero(user_ids_to_sample == user_id)
            num_to_sample = min(
                num_negative_samples * num_positive_samples, len(negative_indices)
            )

            sampled_indices = torch.randperm(len(negative_indices))[:num_to_sample]
            negative_sampled_indices_for_user = negative_indices[sampled_indices]

            negative_samples_indices.append(negative_sampled_indices_for_user)
        negative_sampled_indices = torch.concat(negative_samples_indices)

        tensors = negative_dataset[negative_sampled_indices]
        # return torch.utils.data.TensorDataset(*tensors)
        return tensors

    def extract_sympy_rules(
        self,
        threshold: float = 0.1,
        get_last_layer_rules: bool = False,
        use_cumulative_threshold=False,
    ):
        assert hasattr(self, "atom_names"), (
            "Model must be fitted before extracting rules"
        )

        if not get_last_layer_rules:
            return self._r_model_rules_to_sympy(
                len(self.layers) - 1,
                0,
                threshold,
                use_cumulative_threshold=use_cumulative_threshold,
            )

        # get the input clauses of the last layer, with the weights as tuples
        last_layer_rules = []
        last_layer = self.layers[len(self.layers) - 1]
        last_node_params = last_layer.weights[0]
        last_node_weights = last_layer.activation_function(last_node_params)
        for input_idx, weight in enumerate(last_node_weights):
            weight = weight.item()
            if abs(weight) <= threshold:
                continue

            input_rule = self._r_model_rules_to_sympy(
                len(self.layers) - 2,
                input_idx,
                (threshold / abs(weight) if use_cumulative_threshold else threshold),
                use_cumulative_threshold=use_cumulative_threshold,
            )
            if (
                type(input_rule) is sympy.logic.boolalg.BooleanFalse
                or type(input_rule) is sympy.logic.boolalg.BooleanTrue
            ):
                continue
            if weight > 0:
                last_layer_rules.append(
                    (
                        input_rule,
                        weight,
                    )
                )
            else:
                last_layer_rules.append(
                    (
                        sympy.Not(input_rule),
                        abs(weight),
                    )
                )
        return last_layer_rules

    # recursive
    def _r_model_rules_to_sympy(
        self, layer: int, node: int, threshold: float, use_cumulative_threshold=False
    ):
        assert 0 <= layer < len(self.layers)
        assert 0 <= node < self.layers[layer].weights.shape[0]

        node_weights = self.layers[layer].weights[node]
        clause_inputs = []
        for input_idx, weight in enumerate(node_weights):
            weight = self.layers[layer].activation_function(weight).item()
            if abs(weight) > threshold:
                if layer == 0:
                    clause_inputs.append(
                        sympy.Symbol(self.atom_names[input_idx].replace(" _", "-"))
                    )
                else:
                    adjusted_threshold = threshold
                    if use_cumulative_threshold:
                        # adjust threshold based on number of layers
                        adjusted_threshold = threshold / abs(weight)
                    clause_inputs.append(
                        self._r_model_rules_to_sympy(
                            layer - 1,
                            input_idx,
                            adjusted_threshold,
                            use_cumulative_threshold=use_cumulative_threshold,
                        )
                    )
                if weight < 0:
                    clause_inputs[-1] = sympy.Not(clause_inputs[-1])
        if self.layers[layer].logic_operation == "and":
            return sympy.And(*clause_inputs)
        elif self.layers[layer].logic_operation == "or":
            return sympy.Or(*clause_inputs)
        else:
            raise ValueError(
                f"Unknown logic operation: {self.layers[layer].logic_operation}"
            )

    def fit(
        self,
        train_graph: HeteroData,
        evaluation_graph: HeteroData,
        learning_rate: float,
        loss_fn,
        l1_lambda,
        num_epochs,
        num_negative_samples_per_positive=4,
        optimizer_name="AdamW",
        batch_size=-1,
        device=torch.device("cpu"),
        epoch_checkpoint_prefix: str = "model_epoch_",
    ):
        param_group = [{"params": self.layers.parameters(), "lr": learning_rate}]

        optimizer = create_optimizer(optimizer_name, param_group)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        edge_name = self.config.data.rating_edge_name

        # move model to device
        self = self.to(device)

        # FIXME: one-hot-loss does not converge! (implementation issues)
        self.atom_names = (
            train_graph[edge_name].edge_label_predefined_names
            + [f"u{u}" for u in range(self.config.model.num_user_embedding_atoms)]
            + [f"i{i}" for i in range(self.config.model.num_item_embedding_atoms)]
        )

        # create tensor dataset from predefined atoms and edge labels

        train_dataset = MyDataset(
            graph_data=train_graph,
            negative_sampling_ratio=num_negative_samples_per_positive,
            with_labels=True,
            rating_edge_name=edge_name,
            # TODO: instead just regenerate the dataset after an epoch (self.config.training.perform_repeated_sampling)
        )
        val_dataset = MyDataset(
            graph_data=evaluation_graph,
            with_labels=True,
            rating_edge_name=edge_name,
        )

        train_prediction_losses = []
        val_prediction_losses = []

        epoch_iter = tqdm(range(num_epochs), desc="Training Epochs", unit="epoch")
        for epoch in epoch_iter:
            # Training phase
            self.train()

            l1_penalty_factor = l1_lambda / (
                sum((p.numel() for p in self.layers.parameters() if p.requires_grad))
            )

            orthogonal_lambda = self.config.training.orthogonal_lambda / (
                sum((p.numel() for p in self.layers.parameters() if p.requires_grad))
            )

            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                drop_last=False,
                shuffle=True,
            )

            actual_percent_relevant = []
            expected_percent_relevant = []
            batch_prediction_losses = []

            batch_iter = tqdm(
                dataloader, desc="Train Batches", unit="batch", leave=False
            )
            for predefined_atoms, user_ids, item_ids, labels in batch_iter:
                predefined_atoms = predefined_atoms.to(device)
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                labels = labels.to(device)
                # predict
                # FIXME: use actual user_ids, not the edge indices, which are mapped to a different range!
                y_pred = self(predefined_atoms, user_ids, item_ids)

                actual_percent_relevant.append(torch.mean(y_pred).item())
                expected_percent_relevant.append(
                    torch.sum(labels).item() / labels.shape[0]
                )

                train_prediction_loss = loss_fn(y_pred, labels)
                assert train_prediction_loss.shape == torch.Size([labels.shape[0]]), (
                    f"Expected `reduce=False` on loss function {loss_fn}, but got {train_prediction_loss.shape} instead of {labels.shape}"
                )
                batch_prediction_losses.append(train_prediction_loss.mean().item())
                train_prediction_loss = train_prediction_loss.mean()

                # add l1 regularisation for each variable -> concise rules
                l1_loss = self._compute_sparsity_loss()
                loss = train_prediction_loss + l1_penalty_factor * l1_loss

                if orthogonal_lambda > 0:
                    loss = loss + orthogonal_lambda * self._compute_orthogonality_loss()

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(
                #     self.parameters(), max_norm=1.0, norm_type=2.0
                # )
                optimizer.step()
                batch_iter.set_postfix(
                    {
                        "min(pred)": f"{y_pred.min().item():.4f}",
                        "max(pred)": f"{y_pred.max().item():.4f}",
                        "range(pred)": f"{(y_pred.max().item() - y_pred.min().item()):.4f}",
                        "acc": f"{((y_pred > 0.5) == (labels > 0.5)).float().mean().item():.4f}",
                    }
                )

            if (
                self.config.training.save_model_every_n_epochs
                and epoch % self.config.training.save_model_every_n_epochs == 0
            ):
                path = os.path.join(
                    self.config.results_path, f"{epoch_checkpoint_prefix}{epoch}.pth"
                )
                self.save_model(path)

            # Evaluate and reduce learning rate once per epoch
            scheduler.step()
            train_prediction_losses.append(np.mean(batch_prediction_losses))

            if self.config.training.skip_validation:
                continue

            with torch.no_grad():
                # Validation phase
                self.eval()
                # val_dataset = self.create_dataset_for_graph(evaluation_graph)

                val_dataloader = self.create_dataloader_for_graph(
                    val_dataset, batch_size=batch_size, with_labels=True
                )
                y_pred = self.forward_dataloader(
                    val_dataloader, with_labels=True, progress_bar_desc="Val Batches"
                )

                labels = val_dataset.all_labels.to(device)

                val_prediction_loss = loss_fn(y_pred, labels)
                val_prediction_losses.append(val_prediction_loss.mean().item())

            rule_complexity_at_threshold = {}
            for threshold in [0.5, 0.55]:
                rules = self.extract_sympy_rules(threshold=threshold)
                rule_complexity_at_threshold[f"RC@{threshold}"] = rule_complexity(rules)
            # TODO: evaluate model and show metrics
            epoch_iter.set_postfix(
                {
                    r"mean(pred)": f"{np.mean(actual_percent_relevant):.4f}",
                    r"mean(target)": f"{np.mean(expected_percent_relevant):.4f}",
                    "L_train": f"{np.mean(train_prediction_losses):.4f}",
                    "L_val": f"{np.mean(val_prediction_losses):.4f}",
                    **rule_complexity_at_threshold,
                }
            )

        return train_prediction_losses, val_prediction_losses

    @torch.no_grad()
    def save_model(self, path: str):
        logging.info(f"Saving model to {path}")
        # store the model
        torch.save(self, path)
