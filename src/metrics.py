import torch
import numpy as np
from ebrec.evaluation import (
    MetricEvaluator,
    AucScore,
    NdcgScore,
    MrrScore,
)

from src.utils import group_predictions_by_impression


def evaluation_ranked_metrics(
    impression_ids: list[int], predictions: list[float], labels: list[int]
):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy().tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy().tolist()
    if isinstance(impression_ids, torch.Tensor):
        impression_ids = impression_ids.detach().cpu().numpy().tolist()
    assert len(impression_ids) == len(predictions) == len(labels), (
        "Length of impression_ids, predictions and labels must be the same!"
    )

    impression_ids, predictions, labels = (
        np.array(impression_ids),
        np.array(predictions),
        np.array(labels),
    )

    grouped_predictions = group_predictions_by_impression(
        impression_ids, predictions, labels
    )
    evaluator = MetricEvaluator(
        labels=grouped_predictions["labels"],
        predictions=grouped_predictions["scores"],
        metric_functions=[
            AucScore(),
            MrrScore(),
            NdcgScore(k=5),
            NdcgScore(k=10),
        ],
    )
    evaluator.evaluate()
    return evaluator.evaluations
