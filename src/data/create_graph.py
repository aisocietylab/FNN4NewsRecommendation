from typing import Iterable, Literal
import ast
from collections import Counter
import copy
import itertools
import pickle
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from pathlib import Path
import logging

import scipy
import datasets
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MultiLabelBinarizer
import torch
import torch_geometric
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    import src


TIMESTAMP_UNIT = "us"
TIMESTAMP_DTYPE = f"datetime64[{TIMESTAMP_UNIT}]"

COLUMN_MAPPINGS_BEHAVIORS_EBNERD = {
    "device_type": {
        0: np.nan,
        1: "desktop",
        2: "mobile",
        3: "tablet",
    },
    "postcode": {
        np.nan: np.nan,
        0: "metropolitan",
        1: "rural district",
        2: "municipality",
        3: "provincial",
        4: "big city",
    },
    "is_sso_user": {
        False: "no",
        True: "yes",
    },
    "gender": {
        np.nan: np.nan,
        0: "male",
        1: "female",
    },
    "age": {
        np.nan: np.nan,
        0: "0-10",
        10: "10-20",
        20: "20-30",
        30: "30-40",
        40: "40-50",
        50: "50-60",
        60: "60-70",
        70: "70-80",
        80: "80-90",
        90: "90-100",
    },
    "is_subscriber": {
        False: "no",
        True: "yes",
    },
}


@dataclass
class DatasetBoolenizer:
    column_mappings: dict[str, dict[Any, str]] = field(default_factory=dict)
    process_columns: list[str] | None = None
    """Only process these columns"""
    passthrough_columns: list[str] = field(default_factory=list)
    """These columns are kept without processing"""
    num_keep_frequent_classes: int | None = 20

    # Sklearn preprocessors
    numerical_discretizer: KBinsDiscretizer = field(
        default_factory=lambda: KBinsDiscretizer(
            n_bins=3, encode="onehot", strategy="quantile", subsample=None
        )
    )
    multilabel_binarizer: MultiLabelBinarizer = field(
        default_factory=lambda: MultiLabelBinarizer(sparse_output=True)
    )
    one_hot_encoder: OneHotEncoder | None = None

    # processors fitted on data
    _numerical_column_discretizers: list[KBinsDiscretizer] | None = field(
        init=False, default=None
    )
    _multilabel_column_binarizers: list[MultiLabelBinarizer] | None = field(
        init=False, default=None
    )
    _multilabel_column_names: list[str] | None = field(init=False, default=None)
    fitted: bool = field(init=False, default=False)

    def _create_default_one_hot_encoder(self, categorical_df: pd.DataFrame):
        categories = "auto"
        return OneHotEncoder(
            categories=categories,
            handle_unknown="infrequent_if_exist",
            sparse_output=True,
            max_categories=self.num_keep_frequent_classes,
        )

    def _drop_non_process_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.drop(columns=set(df.columns) - set(self.process_columns))
            if self.process_columns
            else df.drop(columns=set(df.columns))
        )

    def _get_multilabel_column_names(self, df: pd.DataFrame) -> list[str]:
        is_multilabel_column = df.map(
            lambda x: isinstance(x, Iterable) and not isinstance(x, str)
        ).any(axis=0)
        multilabel_columns = is_multilabel_column[is_multilabel_column].index
        return multilabel_columns

    def _slice_numerical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.select_dtypes(include=[np.number]).columns]
        # sorted to have deterministic order
        return df[sorted(df.columns)]

    def _slice_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        potential_columns = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns

        multilabel_columns = self._get_multilabel_column_names(df[potential_columns])
        categorical_columns = set(potential_columns) - set(multilabel_columns)

        # sorted to have deterministic order
        return df[sorted(categorical_columns)]

    def _slice_multilabel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        potential_columns = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns

        multilabel_columns = self._get_multilabel_column_names(df[potential_columns])

        # sorted to have deterministic order
        return df[sorted(multilabel_columns)]

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # first transform columns according to mappings (some numbers, should be mapped to categorical!)
        for column, mapping in self.column_mappings.items():
            df[column] = df[column].map(mapping)
        return df

    def _fit_numerical(self, numerical_df: pd.DataFrame):
        assert self._numerical_column_discretizers is None, (
            "Boolenizer is already fitted!"
        )
        self._numerical_column_discretizers = [
            copy.deepcopy(self.numerical_discretizer) for _ in numerical_df.columns
        ]

        for numerical_col, discretizer in zip(
            numerical_df.columns, self._numerical_column_discretizers
        ):
            col_data = numerical_df[[numerical_col]]
            # remove NaNs (NaNs are mapped to all-NaN atoms, which are later imputed based on a strategy)
            col_data = col_data.dropna()

            discretizer.fit(col_data)
        return self

    def _transform_numerical(self, numerical_df: pd.DataFrame):
        if len(numerical_df.columns) == 0:
            return scipy.sparse.dok_matrix((len(numerical_df), 0))
        numerical_df = numerical_df.copy()

        transformed_columns = []
        for numerical_col, discretizer in zip(
            numerical_df.columns,
            self._numerical_column_discretizers,
        ):
            col_data = numerical_df[[numerical_col]]

            is_nan = col_data.isna()[numerical_col].values

            transformed_data = scipy.sparse.dok_matrix(
                (col_data.shape[0], discretizer.n_bins_[0])
            )
            # only non-NaN values are transformed
            discretized_data = discretizer.transform(col_data[~is_nan])
            transformed_data[~is_nan] = discretized_data
            transformed_data[is_nan] = np.nan

            transformed_columns.append(transformed_data)

        return scipy.sparse.hstack(transformed_columns)

    def _fit_multilabel(self, multilabel_df: pd.DataFrame):
        assert self._multilabel_column_binarizers is None
        self._multilabel_column_binarizers = [
            copy.deepcopy(self.multilabel_binarizer) for _ in multilabel_df.columns
        ]

        self._multilabel_column_names = multilabel_df.columns
        for multilabel_col, binarizer in zip(
            multilabel_df.columns, self._multilabel_column_binarizers
        ):
            binarizer.fit(multilabel_df[multilabel_col])
        return self

    def _transform_multilabel(self, multilabel_df: pd.DataFrame):
        if len(multilabel_df.columns) == 0:
            return scipy.sparse.dok_matrix((len(multilabel_df), 0))
        results = []
        for multilabel_col, binarizer in zip(
            multilabel_df.columns, self._multilabel_column_binarizers
        ):
            results.append(binarizer.transform(multilabel_df[multilabel_col]))
        return scipy.sparse.hstack(results)

    def _map_entities_to_multilabel(self, df: pd.DataFrame) -> pd.DataFrame:
        entity_columns = [col for col in df.columns if col.endswith("_entities")]
        for col in entity_columns:
            # entities are of type `str`, but actually represent literals of type `list[dict]`.
            # We extract the `"Label"` ignoring the rest
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if x else [])
            df[col] = df[col].apply(lambda x: set((d["Label"] for d in x)))
            counter = Counter(itertools.chain.from_iterable(df[col]))
            keep_instances = counter.most_common(self.num_keep_frequent_classes)

            # keep only top-k entities
            keep_set = set([k for k, v in keep_instances])
            df[col] = df[col].apply(lambda x: x.intersection(keep_set))
        return df

    def fit(self, df: pd.DataFrame) -> "DatasetBoolenizer":
        assert not self.fitted, "Boolenizer is already fitted!"
        df = self._drop_non_process_columns(df)
        df = self._map_columns(df)
        df = self._map_entities_to_multilabel(df)

        categorical_df = self._slice_categorical_data(df)
        # encode categorical values
        if self.one_hot_encoder is None:
            # default one hot encoder
            self.one_hot_encoder = self._create_default_one_hot_encoder(categorical_df)

        if categorical_df.isna().sum().sum() > 0:
            logging.debug("Dropping NaNs for fitting one-hot encoder.")
        # NaN will not be treated as a seperate category, instead we use imputation later on
        dropped_categorical_df = categorical_df.dropna()
        self.one_hot_encoder.fit(dropped_categorical_df)

        # numerical data
        numerical_df = self._slice_numerical_data(df)
        self._fit_numerical(numerical_df)

        # multilabel data
        multilabel_df = self._slice_multilabel_data(df)
        assert multilabel_df.isna().sum().sum() == 0, "NaNs found in multilabel data!"
        self._fit_multilabel(multilabel_df)

        self.fitted = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "Call fit or fit_transform first!"
        assert self.one_hot_encoder is not None, "Call fit or fit_transform first!"
        process_df = self._drop_non_process_columns(df)
        process_df = self._map_columns(process_df)
        process_df = self._map_entities_to_multilabel(process_df)

        # categorical data
        categorical_df = self._slice_categorical_data(process_df)
        nan_mask = categorical_df.isna()
        categorical_enc = self.one_hot_encoder.transform(categorical_df)
        # dok is more efficient for setting values
        categorical_enc = categorical_enc.todok()

        if len(categorical_df.columns) > 0:
            # write nans for all one-hot encoded columns where the original value was nan
            output_column_offset = 0
            for column, one_hot_categories in zip(
                self.one_hot_encoder.feature_names_in_, self.one_hot_encoder.categories_
            ):
                new_one_hot_column_offset = output_column_offset + len(
                    one_hot_categories
                )
                categorical_enc[
                    nan_mask[column].values,
                    output_column_offset:new_one_hot_column_offset,
                ] = np.nan
                output_column_offset = new_one_hot_column_offset
            categorical_enc = categorical_enc.tocsr()

        # numerical data
        numerical_df = self._slice_numerical_data(process_df)
        numerical_enc = self._transform_numerical(numerical_df)

        # multilabel data
        multilabel_df = self._slice_multilabel_data(process_df)
        multilabel_enc = self._transform_multilabel(multilabel_df)

        return scipy.sparse.hstack(
            [
                categorical_enc,
                numerical_enc,
                multilabel_enc,
                df[self.passthrough_columns].values,
            ],
            format="csr",
            dtype=np.float32,
        )

    def get_transformed_column_names(self) -> list[str]:
        assert self.one_hot_encoder is not None, "Call fit or fit_transform first!"
        column_names = []

        column_names.extend(self.one_hot_encoder.get_feature_names_out())

        for discretizer in self._numerical_column_discretizers:
            assert len(discretizer.feature_names_in_) == 1
            column_name = discretizer.feature_names_in_[0]
            num_quantiles = discretizer.n_bins_[0]

            percentages = [i * 100 // num_quantiles for i in range(num_quantiles + 1)]
            column_names.extend(
                [
                    f"{column_name}_{start}%-{end}%"
                    for start, end in zip(percentages[:-1], percentages[1:])
                ]
            )
        for column_name, binarizer in zip(
            self._multilabel_column_names, self._multilabel_column_binarizers
        ):
            column_names.extend([f"{column_name}_{c}" for c in binarizer.classes_])
        column_names.extend(self.passthrough_columns)
        return list(column_names)


def create_click_graph_data_ebnerd(
    behaviors_df: pd.DataFrame,
):
    if "article_ids_clicked" in behaviors_df.columns:
        logging.info("Creating labeled dataset")

        positive_impressions = behaviors_df[
            ["user_id", "impression_time", "article_ids_clicked"]
        ].explode(["article_ids_clicked"])

        clicked_df = pd.DataFrame(
            {
                "impression_id": positive_impressions.index.values,
                "user_id": positive_impressions["user_id"].values,
                "article_id": positive_impressions["article_ids_clicked"].values,
                "impression_time": positive_impressions["impression_time"].astype(
                    TIMESTAMP_DTYPE
                ),
                "label": np.ones(len(positive_impressions), dtype=np.bool_),
            }
        )
        negative_impressions = behaviors_df[
            [
                "user_id",
                "impression_time",
                "article_ids_clicked",
                "article_ids_inview",
            ]
        ].copy()
        negative_impressions["article_ids_not_clicked"] = negative_impressions[
            ["article_ids_clicked", "article_ids_inview"]
        ].apply(
            lambda x: set(x["article_ids_inview"]) - set(x["article_ids_clicked"]),
            axis=1,
        )
        negative_impressions = negative_impressions.drop(
            columns=["article_ids_clicked", "article_ids_inview"]
        )
        negative_impressions = negative_impressions.explode("article_ids_not_clicked")
        not_clicked_df = pd.DataFrame(
            {
                "impression_id": negative_impressions.index.values,
                "user_id": negative_impressions["user_id"].values,
                "article_id": negative_impressions["article_ids_not_clicked"].values,
                "impression_time": negative_impressions["impression_time"].astype(
                    TIMESTAMP_DTYPE
                ),
                "label": np.zeros(len(negative_impressions), dtype=np.bool_),
            }
        )

        graph_data_df = pd.concat([clicked_df, not_clicked_df], ignore_index=True)
    else:
        logging.info("Creating unlabeled dataset")
        impressions = behaviors_df[
            [
                "user_id",
                "impression_time",
                "article_ids_inview",
            ]
        ].explode("article_ids_inview")

        graph_data_df = pd.DataFrame(
            {
                "impression_id": impressions.index.values,
                "user_id": impressions["user_id"].values,
                "article_id": impressions["article_ids_inview"].values,
                "impression_time": impressions["impression_time"].astype(
                    TIMESTAMP_DTYPE
                ),
            }
        )
    return graph_data_df


def create_click_graph_data_mind(
    behaviors_df: pd.DataFrame,
):
    if "impressions" not in behaviors_df.columns:
        raise ValueError("Support for the test dataset not implemented yet!")
    impressions = behaviors_df[["user_id", "time", "impressions"]].copy()

    impressions["impressions"] = impressions["impressions"].str.split(" ")
    impressions = impressions.explode("impressions")

    clicked_suffix = "-1"
    clicked_mask = impressions["impressions"].str.endswith(clicked_suffix)

    impressions["impressions"] = (
        impressions["impressions"].str.rstrip(clicked_suffix).str.rstrip("-0")
    )

    return pd.DataFrame(
        {
            "impression_id": impressions.index.values,
            "user_id": impressions["user_id"].values,
            "article_id": impressions["impressions"].values,
            "impression_time": impressions["time"].astype(TIMESTAMP_DTYPE),
            "label": clicked_mask.astype(np.bool_),
        }
    )


def create_article_age_atoms(article_age: pd.Series) -> pd.DataFrame:
    age_atoms = pd.DataFrame(
        {
            "ArticleAge<30m": article_age < pd.Timedelta(minutes=30),
            "30m≤ArticleAge<1h": (pd.Timedelta(minutes=30) <= article_age)
            & (article_age < pd.Timedelta(hours=1)),
            "1h≤ArticleAge<2h": (pd.Timedelta(hours=1) <= article_age)
            & (article_age < pd.Timedelta(hours=2)),
            "2h≤ArticleAge<4h": (pd.Timedelta(hours=2) <= article_age)
            & (article_age < pd.Timedelta(hours=4)),
            "4h≤ArticleAge<8h": (pd.Timedelta(hours=4) <= article_age)
            & (article_age < pd.Timedelta(hours=8)),
            "8h≤ArticleAge<12h": (pd.Timedelta(hours=8) <= article_age)
            & (article_age < pd.Timedelta(hours=12)),
            "12h≤ArticleAge<1d": (pd.Timedelta(hours=12) <= article_age)
            & (article_age < pd.Timedelta(days=1)),
            "1d≤ArticleAge<2d": (pd.Timedelta(days=1) <= article_age)
            & (article_age < pd.Timedelta(days=2)),
            "2d≤ArticleAge<3d": (pd.Timedelta(days=2) <= article_age)
            & (article_age < pd.Timedelta(days=3)),
            "3d≤ArticleAge": pd.Timedelta(days=3) <= article_age,
        },
        dtype=np.float32,
    )
    # keep NaNs as NaNs for imputer!
    age_atoms[np.isnan(article_age)] = np.nan
    return age_atoms


def create_graph_for_split(
    behaviors_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    behavior_boolenizer: DatasetBoolenizer,
    article_boolenizer: DatasetBoolenizer,
    add_article_age_atoms: bool,
):
    assert behaviors_df.index.name == "impression_id"
    if "article_ids_inview" in behaviors_df.columns:
        logging.info("Creating graph for EB-NeRD dataset")
        graph_data_df = create_click_graph_data_ebnerd(behaviors_df)
    else:
        logging.info("Creating graph for MIND dataset")
        graph_data_df = create_click_graph_data_mind(behaviors_df)

    # create graph
    data = torch_geometric.data.HeteroData()

    users_df = behaviors_df.groupby("user_id").first()
    if "article_ids_inview" in behaviors_df.columns:
        users_df = users_df.drop(
            # behavior specific columns, not user specific
            columns=[
                "article_ids_inview",
                "article_ids_clicked",
                "read_time",
                "scroll_percentage",
            ]
        )
    else:
        users_df = users_df.drop(
            columns=[
                "history",
                "impressions",
            ]
        )
    encoded_behaviors = behavior_boolenizer.transform(behaviors_df)
    encoded_articles = article_boolenizer.transform(articles_df)

    unknown_articles = graph_data_df["article_id"][
        ~graph_data_df["article_id"].isin(articles_df.index)
    ].unique()
    if len(unknown_articles) > 0:
        logging.warning(
            f"Some article ids were not found. Adding them with NaN features (still allows us to learn embeddings). {len(unknown_articles)} unknown article ids found."
        )
        articles_df = pd.concat(
            [
                articles_df,
                pd.DataFrame(
                    index=unknown_articles,
                    columns=articles_df.columns,
                    data=np.nan,
                ),
            ]
        )

        encoded_articles = scipy.sparse.vstack(
            [
                encoded_articles,
                scipy.sparse.csr_matrix(
                    (len(unknown_articles), encoded_articles.shape[1])
                ),
            ]
        )

    # Users
    data[
        "user"
    ].x = users_df  # torch.empty((users_df.shape[0], 0), dtype=torch.float32)
    data["user"].num_nodes = len(users_df)
    data["user"].x_names = []
    global_user_ids = users_df.index
    if not pd.api.types.is_any_real_numeric_dtype(global_user_ids):
        # remove `U` prefix from users in MIND dataset
        global_user_ids = global_user_ids.str.strip("U").values
    data["user"].global_id = torch.tensor(global_user_ids.astype(np.int64))

    # Items
    data["item"].x = articles_df  # encoded_articles
    data["item"].num_nodes = len(articles_df)
    data["item"].x_names = article_boolenizer.get_transformed_column_names()
    global_item_ids = articles_df.index
    if not pd.api.types.is_any_real_numeric_dtype(global_item_ids):
        # remove `N` prefix from news articles in MIND dataset
        global_item_ids = global_item_ids.str.strip("N").values
    data["item"].global_id = torch.tensor(global_item_ids.astype(np.int64))

    edge_index = torch.vstack(
        [
            torch.tensor(users_df.index.get_indexer(graph_data_df["user_id"])),
            torch.tensor(articles_df.index.get_indexer(graph_data_df["article_id"])),
        ]
    )
    impression_time = graph_data_df["impression_time"].values

    encoded_behaviors_idx = behaviors_df.index.get_indexer(
        graph_data_df["impression_id"]
    )
    encoded_articles_idx = articles_df.index.get_indexer(graph_data_df["article_id"])
    assert all(encoded_behaviors_idx >= 0), (
        f"Some impression ids were not found. {graph_data_df['impression_id'][encoded_behaviors_idx < 0]}"
    )
    assert all(encoded_articles_idx >= 0), (
        f"Some article ids were not found. {graph_data_df['article_id'][encoded_articles_idx < 0]}"
    )
    encoded_input = scipy.sparse.hstack(
        [
            # FIXME: there is a bug in the scipy sparse matrix indexing implementation, because it downcasts to int32 internally, leading to an overflow during indexing for large datasets
            encoded_behaviors[encoded_behaviors_idx],
            encoded_articles[encoded_articles_idx],
        ],
        format="csr",
    ).tocoo()

    encoded_input_columns = (
        behavior_boolenizer.get_transformed_column_names()
        + article_boolenizer.get_transformed_column_names()
    )
    # source of predefined atoms (user, article or impression (e.g., article age)).
    encoded_input_sources = ["user"] * len(
        behavior_boolenizer.get_transformed_column_names()
    ) + ["article"] * len(article_boolenizer.get_transformed_column_names())

    if add_article_age_atoms:
        article_release_time = articles_df.loc[
            graph_data_df["article_id"], "published_time"
        ].values
        article_age = impression_time - article_release_time
        article_age_atoms = create_article_age_atoms(article_age)

        encoded_input = scipy.sparse.hstack(
            [encoded_input, article_age_atoms.values.astype(np.float32)],
        )
        encoded_input_columns.extend(article_age_atoms.columns)
        encoded_input_sources.extend(["impression"] * len(article_age_atoms.columns))

    # TODO: support sparse tensors with the rest of the framework
    encoded_input = torch.sparse_coo_tensor(
        np.vstack((encoded_input.row, encoded_input.col)),
        encoded_input.data,
        encoded_input.shape,
    ).to_dense()

    impression_id = torch.tensor(graph_data_df["impression_id"].values.astype(np.int64))

    data["user", "rates", "item"].edge_index = edge_index
    data["user", "rates", "item"].time = impression_time
    data["user", "rates", "item"].x = encoded_input
    data["user", "rates", "item"].global_id = impression_id

    data["user", "rates", "item"].edge_label_index = edge_index
    data["user", "rates", "item"].edge_label_time = impression_time
    data["user", "rates", "item"].edge_label_predefined = encoded_input
    data["user", "rates", "item"].edge_label_global_id = impression_id
    data["user", "rates", "item"].edge_label_predefined_names = encoded_input_columns
    data["user", "rates", "item"].edge_label_predefined_source = encoded_input_sources
    if "label" in graph_data_df:
        data["user", "rates", "item"].edge_label = torch.tensor(
            graph_data_df["label"].values, dtype=torch.float32
        )

    data.validate()
    return data


def add_feature_names(
    data: torch_geometric.data.HeteroData,
    article_feature_names: list[str],
    behavior_feature_names: list[str],
):
    data["item"].x_names = article_feature_names
    # data["user"].x_names = behavior_feature_names
    data["user"].x_names = []
    # NOTE: needs to align with how we stacked the features!
    data["user", "rates", "item"].edge_label_predefined_names = (
        article_feature_names + behavior_feature_names
    )
    data.validate()
    return data


def sample_fraction(dataset: datasets.DatasetDict, fraction: float, seed: int = 42):
    logging.info(f"Using only {fraction * 100}% of the dataset for training.")
    dataset["train"] = (
        dataset["train"]
        .shuffle(seed=seed)
        .select(range(int(len(dataset["train"]) * fraction)))
    )
    dataset["val"] = (
        dataset["val"]
        .shuffle(seed=seed)
        .select(range(int(len(dataset["val"]) * fraction)))
    )
    return dataset


def impute_missing_values(
    train_df: HeteroData,
    val_df: HeteroData,
    test_df: HeteroData | None = None,
    strategy: Literal["mean", "median", "false", "0.5", "true"] = "mean",
    imputer: ColumnTransformer | None = None,
):
    if strategy in ["mean", "median"]:
        fill_value = None
    else:
        if strategy == "false":
            fill_value = 0
        elif strategy == "0.5":
            fill_value = 0.5
        elif strategy == "true":
            fill_value = 1
        strategy = "constant"
    encoded_input_sources = train_df[
        "user", "rates", "item"
    ].edge_label_predefined_source
    assert all(s in ["user", "article", "impression"] for s in encoded_input_sources)
    user_ids, article_ids = train_df["user", "rates", "item"].edge_label_index

    user_indices = np.array(
        [i for i, s in enumerate(encoded_input_sources) if s == "user"]
    )
    article_indices = np.array(
        [i for i, s in enumerate(encoded_input_sources) if s == "article"]
    )
    impression_indices = np.array(
        [i for i, s in enumerate(encoded_input_sources) if s == "impression"]
    )

    def deduplicate_features_by_id(data: torch.Tensor, ids: torch.Tensor):
        unique_ids, group_ids = torch.unique(
            ids, return_inverse=True
        )  # return_inverse gives us the mapping from original indices to unique indices, which can be viewed as group ids
        unique_impressions = torch.empty(
            (len(unique_ids), data.shape[1]), dtype=torch.float32
        )
        # write to the same indices will just overwrite, so we get each article/user only once
        unique_impressions[group_ids] = data.float()
        return unique_impressions

    # Get each users unique features
    deduplicated_user_features = deduplicate_features_by_id(
        train_df["user", "rates", "item"].edge_label_predefined[:, user_indices],
        user_ids,
    )

    # only index an article once
    deduplicated_article_features = deduplicate_features_by_id(
        train_df["user", "rates", "item"].edge_label_predefined[:, article_indices],
        article_ids,
    )

    if imputer is None:
        # we need to group `user` and `article` features, so we don't estimate the mean/median over the exploded impressions. The mean and median would be biased towards articles and users with many impressions.
        user_imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        article_imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        impression_imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

        transformers = []
        if len(user_indices) > 0:
            transformers.append(("user", user_imputer, user_indices))
        if len(article_indices) > 0:
            transformers.append(("article", article_imputer, article_indices))
        if len(impression_indices) > 0:
            transformers.append(("impression", impression_imputer, impression_indices))

        imputer = ColumnTransformer(
            transformers,
            sparse_threshold=0,  # always return dense, because we use pytorch tensors (sparse is not yet fully supported)
        )
        imputer.fit(
            train_df["user", "rates", "item"].edge_label_predefined,
        )

        if len(user_indices) > 0:
            user_imputer.fit(deduplicated_user_features)
        if len(article_indices) > 0:
            article_imputer.fit(deduplicated_article_features)
        if len(impression_indices) > 0:
            impression_imputer.fit(
                train_df["user", "rates", "item"].edge_label_predefined[
                    :, impression_indices
                ]
            )
    train_df["user", "rates", "item"].edge_label_predefined = torch.tensor(
        imputer.transform(train_df["user", "rates", "item"].edge_label_predefined)
    )
    val_df["user", "rates", "item"].edge_label_predefined = torch.tensor(
        imputer.transform(val_df["user", "rates", "item"].edge_label_predefined)
    )
    if test_df is not None:
        test_df["user", "rates", "item"].edge_label_predefined = torch.tensor(
            imputer.transform(test_df["user", "rates", "item"].edge_label_predefined)
        )
    return (
        imputer,
        train_df,
        val_df,
        test_df,
    )


def _load_ebnerd_data(
    dataset_base_path: Path | str, test_path: Path | str | None = None
):
    dataset_base_path = Path(dataset_base_path)

    data_files_articles = {"train": str(dataset_base_path / "articles.parquet")}
    data_files_behaviors = {
        "train": str(dataset_base_path / "train" / "behaviors.parquet"),
        "val": str(dataset_base_path / "validation" / "behaviors.parquet"),
    }
    if test_path is not None:
        data_files_articles["test"] = str(Path(test_path) / "articles.parquet")
        data_files_behaviors["test"] = str(
            Path(test_path) / "test" / "behaviors.parquet"
        )

    articles = datasets.load_dataset("parquet", data_files=data_files_articles)
    behaviors = datasets.load_dataset("parquet", data_files=data_files_behaviors)
    # might leak information
    behaviors = behaviors.remove_columns(
        [
            "session_id",
            "next_read_time",
            "next_scroll_percentage",
        ]
    )
    return articles, behaviors


def create_ebnerd_graph_datasets(
    data_config: "src.DatasetConfig",
    behavior_boolenizer: DatasetBoolenizer | None = None,
    article_boolenizer: DatasetBoolenizer | None = None,
    imputer: ColumnTransformer | None = None,
    seed: int | None = None,
    save_path_boolenizers: str | os.PathLike | None = None,
    impression_ids_to_exclude: set[int] | None = None,
) -> tuple[
    torch_geometric.data.HeteroData,
    torch_geometric.data.HeteroData,
    torch_geometric.data.HeteroData | None,
]:
    dataset_base_path = Path(data_config.base_path)

    articles, behaviors = _load_ebnerd_data(dataset_base_path)

    if data_config.fraction is not None:
        behaviors = sample_fraction(behaviors, data_config.fraction, seed)

    if impression_ids_to_exclude is not None:
        logging.info(
            f"Excluding {len(impression_ids_to_exclude)} impression ids from training and validation data."
        )
        behaviors["train"] = behaviors["train"].filter(
            lambda example: example["impression_id"] not in impression_ids_to_exclude
        )
        behaviors["val"] = behaviors["val"].filter(
            lambda example: example["impression_id"] not in impression_ids_to_exclude
        )
        if "test" in behaviors:
            behaviors["test"] = behaviors["test"].filter(
                lambda example: example["impression_id"]
                not in impression_ids_to_exclude
            )

    if behavior_boolenizer is None:
        behavior_boolenizer = DatasetBoolenizer(
            column_mappings=COLUMN_MAPPINGS_BEHAVIORS_EBNERD,
            process_columns=[
                "read_time",
                "scroll_percentage",
                "device_type",
                "is_sso_user",
                "gender",
                "postcode",
                "age",
                "is_subscriber",
            ],
            num_keep_frequent_classes=data_config.num_keep_frequent_classes,
        )
    if article_boolenizer is None:
        article_boolenizer = DatasetBoolenizer(
            process_columns=[
                "premium",
                "article_type",
                "category_str",
                "sentiment_label",
                # numerical
                "total_inviews",
                "total_pageviews",
                "total_read_time",
                # 78 unique values
                "topics",
                # 172 unique values (just ints, not very explainable)
                "subcategory",
            ],
            passthrough_columns=[
                "sentiment_score",
            ],
            num_keep_frequent_classes=data_config.num_keep_frequent_classes,
        )

    # to pandas
    train_behaviors_df = behaviors["train"].to_pandas().set_index("impression_id")
    val_behaviors_df = behaviors["val"].to_pandas().set_index("impression_id")

    train_articles_df = articles["train"].to_pandas().set_index("article_id")

    # fit on training data
    if not behavior_boolenizer.fitted:
        behavior_boolenizer.fit(train_behaviors_df)

    if not article_boolenizer.fitted:
        article_boolenizer.fit(train_articles_df)

    train_data = create_graph_for_split(
        train_behaviors_df,
        train_articles_df,
        behavior_boolenizer,
        article_boolenizer,
        add_article_age_atoms=data_config.add_article_age_atoms,
    )
    val_data = create_graph_for_split(
        val_behaviors_df,
        train_articles_df,
        behavior_boolenizer,
        article_boolenizer,
        add_article_age_atoms=data_config.add_article_age_atoms,
    )

    if "test" in behaviors:
        assert "test" in articles, (
            "If behaviors has test, articles must also have test!"
        )
        test_articles_df = articles["test"].to_pandas().set_index("article_id")
        test_behaviors_df = behaviors["test"].to_pandas().set_index("impression_id")

        test_data = create_graph_for_split(
            test_behaviors_df,
            test_articles_df,
            behavior_boolenizer,
            article_boolenizer,
            add_article_age_atoms=data_config.add_article_age_atoms,
        )
    else:
        test_data = None

    # impute missing values
    imputer, train_data, val_data, test_data = impute_missing_values(
        train_data, val_data, test_data, strategy=data_config.impute_strategy
    )

    if save_path_boolenizers is not None:
        logging.info(f"Saving boolenizers and imputer to {save_path_boolenizers}")
        save_path = Path(save_path_boolenizers)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "article_boolenizer.pkl", "wb") as f:
            pickle.dump(article_boolenizer, f)
        with open(save_path / "behavior_boolenizer.pkl", "wb") as f:
            pickle.dump(behavior_boolenizer, f)
        with open(save_path / "imputer.pkl", "wb") as f:
            pickle.dump(imputer, f)

    # num atoms
    logging.debug(
        f"Article boolenizer num atoms: {len(article_boolenizer.get_transformed_column_names())}"
    )
    logging.debug(
        f"Behavior boolenizer num atoms: {len(behavior_boolenizer.get_transformed_column_names())}"
    )
    logging.debug(f"{train_data=}")
    logging.debug(f"{val_data=}")
    logging.debug(f"{test_data=}")
    return train_data, val_data, test_data


def _load_mind_data(dataset_base_path: Path | str, test_path: Path | str | None = None):
    dataset_base_path = Path(dataset_base_path)
    data_files_articles = {
        "train": str(dataset_base_path / "train" / "news.tsv"),
        "val": str(dataset_base_path / "valid" / "news.tsv"),
    }
    data_files_behaviors = {
        "train": str(dataset_base_path / "train" / "behaviors.tsv"),
        "val": str(dataset_base_path / "valid" / "behaviors.tsv"),
    }

    if test_path is not None:
        data_files_articles["test"] = str(Path(test_path) / "news.tsv")
        data_files_behaviors["test"] = str(Path(test_path) / "behaviors.tsv")

    articles = datasets.load_dataset(
        "csv",
        data_files=data_files_articles,
        sep="\t",
        column_names=[
            "article_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ],
    )
    behaviors = datasets.load_dataset(
        "csv",
        data_files=data_files_behaviors,
        sep="\t",
        column_names=["impression_id", "user_id", "time", "history", "impressions"],
    )

    return articles, behaviors


def create_mind_graph_datasets(
    data_config: "src.DatasetConfig",
    behavior_boolenizer: DatasetBoolenizer | None = None,
    article_boolenizer: DatasetBoolenizer | None = None,
    imputer: ColumnTransformer | None = None,
    seed: int | None = None,
    save_path_boolenizers: str | os.PathLike | None = None,
) -> tuple[torch_geometric.data.HeteroData, torch_geometric.data.HeteroData]:
    dataset_base_path = Path(data_config.base_path)

    articles, behaviors = _load_mind_data(dataset_base_path)

    if data_config.fraction is not None:
        behaviors = sample_fraction(behaviors, data_config.fraction, seed)

    if behavior_boolenizer is None:
        # no behavior features to boolenize, **but** we estimate the article age at impression times.
        behavior_boolenizer = DatasetBoolenizer(
            num_keep_frequent_classes=data_config.num_keep_frequent_classes
        )
    if article_boolenizer is None:
        article_boolenizer = DatasetBoolenizer(
            process_columns=[
                "category",
                "subcategory",
                "title_entities",
                "abstract_entities",
            ],
            num_keep_frequent_classes=data_config.num_keep_frequent_classes,
        )

    # to pandas
    train_articles_df = articles["train"].to_pandas().set_index("article_id")
    val_articles_df = articles["val"].to_pandas().set_index("article_id")

    train_behaviors_df = behaviors["train"].to_pandas().set_index("impression_id")
    val_behaviors_df = behaviors["val"].to_pandas().set_index("impression_id")
    train_behaviors_df["time"] = pd.to_datetime(train_behaviors_df["time"])
    val_behaviors_df["time"] = pd.to_datetime(val_behaviors_df["time"])

    impression_times = train_behaviors_df[["impressions", "time"]].copy()
    impression_times["impressions"] = impression_times["impressions"].str.split(" ")
    impression_times = impression_times.explode("impressions")
    impression_times["impressions"] = (
        impression_times["impressions"].str.rstrip("-0").str.rstrip("-1")
    )
    impression_times = impression_times.rename(columns={"impressions": "article_id"})
    impression_times = impression_times.groupby("article_id")["time"].min()

    # NOTE: we estimate the article release time as the first impression time it was shown in the training data (similar to how MIND authors estimate the survival time of articles)
    train_articles_df["published_time"] = train_articles_df.join(impression_times)[
        "time"
    ]
    val_articles_df["published_time"] = val_articles_df.join(impression_times)["time"]

    # fit on training data
    if not behavior_boolenizer.fitted:
        behavior_boolenizer.fit(train_behaviors_df)

    if not article_boolenizer.fitted:
        article_boolenizer.fit(train_articles_df)

    train_data = create_graph_for_split(
        train_behaviors_df,
        train_articles_df,
        behavior_boolenizer,
        article_boolenizer,
        add_article_age_atoms=data_config.add_article_age_atoms,
    )
    val_data = create_graph_for_split(
        val_behaviors_df,
        val_articles_df,
        behavior_boolenizer,
        article_boolenizer,
        add_article_age_atoms=data_config.add_article_age_atoms,
    )

    if "test" in behaviors:
        assert "test" in articles, (
            "If behaviors has test, articles must also have test!"
        )
        test_articles_df = articles["test"].to_pandas().set_index("article_id")
        test_behaviors_df = behaviors["test"].to_pandas().set_index("impression_id")

        test_behaviors_df["time"] = pd.to_datetime(test_behaviors_df["time"])
        test_articles_df["published_time"] = test_articles_df.join(impression_times)[
            "time"
        ]

        test_data = create_graph_for_split(
            test_behaviors_df,
            test_articles_df,
            behavior_boolenizer,
            article_boolenizer,
            add_article_age_atoms=data_config.add_article_age_atoms,
        )
    else:
        test_data = None

    imputer, train_data, val_data, test_data = impute_missing_values(
        train_data,
        val_data,
        test_data,
        strategy=data_config.impute_strategy,
        imputer=imputer,
    )

    if save_path_boolenizers is not None:
        logging.info(f"Saving boolenizers and imputer to {save_path_boolenizers}")
        save_path = Path(save_path_boolenizers)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "article_boolenizer.pkl", "wb") as f:
            pickle.dump(article_boolenizer, f)
        with open(save_path / "behavior_boolenizer.pkl", "wb") as f:
            pickle.dump(behavior_boolenizer, f)
        with open(save_path / "imputer.pkl", "wb") as f:
            pickle.dump(imputer, f)

    # num atoms
    logging.debug(
        f"Article boolenizer num atoms: {len(article_boolenizer.get_transformed_column_names())}"
    )
    logging.debug(
        f"Behavior boolenizer num atoms: {len(behavior_boolenizer.get_transformed_column_names())}"
    )
    logging.debug(f"{train_data=}")
    logging.debug(f"{val_data=}")
    logging.debug(f"{test_data=}")
    return train_data, val_data, test_data
