#!/usr/bin/env python3
# Slightly modified version of: https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/quick_start/nrms_ebnerd.py
import json
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import numpy as np
import os
import simple_parsing

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import (
    slice_join_dataframes,
    concat_str_columns,
)
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import rank_predictions_by_score

from ebrec.models.newsrec.dataloader import NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors


@dataclass
class NRMSConfig:
    PATH: Path = Path("datasets").expanduser()
    SEED: int = np.random.randint(0, 1_000)
    DATASPLIT: str = "ebnerd_small"
    EPOCHS: int = 10 
    FRACTION: float = 1.0

    RESULTS_PATH: Path = Path("results").expanduser() / "ebnerd_nrms_metrics.json"

    DUMP_DIR = Path("ebnerd_nrms_predictions")

    MAX_TITLE_LENGTH = 30
    HISTORY_SIZE = 20
    FRACTION_TEST = 1.0

    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VAL = 32
    BATCH_SIZE_TEST_WO_B = 32
    BATCH_SIZE_TEST_W_B = 4
    N_CHUNKS_TEST = 10
    CHUNKS_DONE = 0

    COLUMNS = [
        DEFAULT_USER_COL,
        DEFAULT_HISTORY_ARTICLE_ID_COL,
        DEFAULT_INVIEW_ARTICLES_COL,
        DEFAULT_CLICKED_ARTICLES_COL,
        DEFAULT_IMPRESSION_ID_COL,
    ]


def main(arguments: list[str] | None = None):
    cfg = simple_parsing.parse(
        config_class=NRMSConfig,
        args=arguments,
    )
    cfg.DUMP_DIR.mkdir(exist_ok=True, parents=True)

    MODEL_NAME = f"NRMS-{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-{cfg.SEED}"
    print(f"Dir: {MODEL_NAME}")

    MODEL_WEIGHTS = cfg.DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
    LOG_DIR = cfg.DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")
    TEST_DF_DUMP = cfg.DUMP_DIR.joinpath("test_predictions", MODEL_NAME)
    TEST_DF_DUMP.mkdir(parents=True, exist_ok=True)

    hparams_nrms.history_size = cfg.HISTORY_SIZE

    df_train = (
        ebnerd_from_path(
            cfg.PATH.joinpath(cfg.DATASPLIT, "train"),
            history_size=cfg.HISTORY_SIZE,
        )
        .sample(fraction=cfg.FRACTION)
        .select(cfg.COLUMNS)
        .pipe(
            sampling_strategy_wu2019,
            npratio=4,
            shuffle=True,
            with_replacement=True,
            seed=cfg.SEED,
        )
        .pipe(create_binary_labels_column)
    )
    df_validation = (
        ebnerd_from_path(
            cfg.PATH.joinpath(cfg.DATASPLIT, "validation"),
            history_size=cfg.HISTORY_SIZE,
        )
        .sample(fraction=cfg.FRACTION)
        .select(cfg.COLUMNS)
        .pipe(create_binary_labels_column)
    )

    df_articles = pl.read_parquet(cfg.PATH.joinpath(cfg.DATASPLIT, "articles.parquet"))

    # =>
    TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
    TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]

    # LOAD HUGGINGFACE:
    transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
    transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    word2vec_embedding = get_transformers_word_embeddings(transformer_model)
    #
    df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
    df_articles, token_col_title = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, cat_cal, max_length=cfg.MAX_TITLE_LENGTH
    )
    # =>
    article_mapping = create_article_id_to_value_mapping(
        df=df_articles, value_col=token_col_title
    )

    # =>
    print("Init train- and val-dataloader")
    train_dataloader = NRMSDataLoaderPretransform(
        behaviors=df_train,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False,
        batch_size=cfg.BATCH_SIZE_TRAIN,
    )

    # CALLBACKS
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR, histogram_freq=1
    )
    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_WEIGHTS, save_best_only=True, save_weights_only=True, verbose=1
    )

    model = NRMSModel(
        hparams=hparams_nrms,
        word2vec_embedding=word2vec_embedding,
        seed=42,
    )
    model.model.fit(
        train_dataloader,
        epochs=cfg.EPOCHS,
        callbacks=[tensorboard_callback, modelcheckpoint],
    )
    val_dataloader_predict = NRMSDataLoaderPretransform(
        behaviors=df_validation,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=cfg.BATCH_SIZE_VAL,
    )

    scores = model.scorer.predict(val_dataloader_predict)
    df_validation = add_prediction_scores(df_validation, scores.tolist()).with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )

    metrics = MetricEvaluator(
        labels=df_validation["labels"].to_list(),
        predictions=df_validation["scores"].to_list(),
        metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
    )
    metric_results = metrics.evaluate()
    print(metric_results.evaluations)

    cfg.RESULTS_PATH.write_text(json.dumps(metric_results.evaluations, indent=2))


if __name__ == "__main__":
    main()
