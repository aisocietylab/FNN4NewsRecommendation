# %% [markdown]
# <i>Copyright (c) Recommenders contributors.</i>
#
# <i>Licensed under the MIT License.</i>
# <i>Modified for FNN4NewsRecommendation Submission</i>

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import pickle
from collections import Counter
from recommenders.datasets.mind import (
    download_and_extract_glove,
    load_glove_matrix,
    word_tokenize,
)
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set
from recommenders.utils.notebook_utils import store_metadata

from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

tf.get_logger().setLevel("ERROR")  # only show error messages
print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))


seed = 42
batch_size = 32
# word_embedding_dim should be in [50, 100, 200, 300]
word_embedding_dim = 300

# MIND_type = "demo"
MIND_type = "small"

# data_path = tmpdir.name
data_path = f"datasets/mind_{MIND_type}"

train_news_file = os.path.join(data_path, "train", r"news.tsv")
train_behaviors_file = os.path.join(data_path, "train", r"behaviors.tsv")
valid_news_file = os.path.join(data_path, "valid", r"news.tsv")
valid_behaviors_file = os.path.join(data_path, "valid", r"behaviors.tsv")
yaml_file = os.path.join("configs", "nrms_mind.yaml")

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(
    MIND_type
)

# %% Prepare embeddings and dictionary files
# Adapted from: https://github.com/recommenders-team/recommenders/blob/276d3aa55e2af87b588327e56812d774c8e3fe8c/examples/01_prepare_data/mind_utils.ipynb

tmpdir = TemporaryDirectory()
output_path = os.path.join(tmpdir.name, "utils")
os.makedirs(output_path, exist_ok=True)

news = pd.read_table(
    os.path.join(data_path, "train", "news.tsv"),
    names=[
        "newid",
        "vertical",
        "subvertical",
        "title",
        "abstract",
        "url",
        "entities in title",
        "entities in abstract",
    ],
    usecols=["vertical", "subvertical", "title", "abstract"],
)


news_vertical = news.vertical.drop_duplicates().reset_index(drop=True)
vert_dict_inv = news_vertical.to_dict()
vert_dict = {v: k + 1 for k, v in vert_dict_inv.items()}

news_subvertical = news.subvertical.drop_duplicates().reset_index(drop=True)
subvert_dict_inv = news_subvertical.to_dict()
subvert_dict = {v: k + 1 for k, v in vert_dict_inv.items()}

news.title = news.title.apply(word_tokenize)
news.abstract = news.abstract.apply(word_tokenize)
word_cnt = Counter()
word_cnt_all = Counter()

for i in tqdm(range(len(news))):
    word_cnt.update(news.loc[i]["title"])
    word_cnt_all.update(news.loc[i]["title"])
    word_cnt_all.update(news.loc[i]["abstract"])

word_dict = {k: v + 1 for k, v in zip(word_cnt, range(len(word_cnt)))}
word_dict_all = {k: v + 1 for k, v in zip(word_cnt_all, range(len(word_cnt_all)))}

wordDict_file = os.path.join(output_path, "word_dict.pkl")
with open(wordDict_file, "wb") as f:
    pickle.dump(word_dict, f)

glove_path = download_and_extract_glove(data_path)

embedding_matrix, exist_word = load_glove_matrix(
    glove_path, word_dict, word_embedding_dim
)
embedding_all_matrix, exist_all_word = load_glove_matrix(
    glove_path, word_dict_all, word_embedding_dim
)


wordEmb_file = os.path.join(output_path, "embedding.npy")
np.save(wordEmb_file, embedding_matrix)

uid2index = {}
with open(os.path.join(data_path, "train", "behaviors.tsv"), "r") as f:
    for line in tqdm(f):
        uid = line.strip("\n").split("\t")[1]
        if uid not in uid2index:
            uid2index[uid] = len(uid2index) + 1

userDict_file = os.path.join(output_path, "uid2index.pkl")
with open(userDict_file, "wb") as f:
    pickle.dump(uid2index, f)

hparams = prepare_hparams(
    yaml_file,
    wordEmb_file=wordEmb_file,
    wordDict_file=wordDict_file,
    userDict_file=userDict_file,
    batch_size=batch_size,
    show_step=10,
)
print(hparams)

iterator = MINDIterator

model = NRMSModel(hparams, iterator, seed=seed)
print("Init. performance:", model.run_eval(valid_news_file, valid_behaviors_file))

model.fit(
    train_news_file,
    train_behaviors_file,
    valid_news_file,
    valid_behaviors_file,
)

res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
print("Old eval:", res_syn)

store_metadata("group_auc", res_syn["group_auc"])
store_metadata("mean_mrr", res_syn["mean_mrr"])
store_metadata("ndcg@5", res_syn["ndcg@5"])
store_metadata("ndcg@10", res_syn["ndcg@10"])

if model.support_quick_scoring:
    group_impr_indexes, group_labels, group_preds = model.run_fast_eval(
        valid_news_file, valid_behaviors_file
    )
else:
    group_impr_indexes, group_labels, group_preds = model.run_slow_eval(
        valid_news_file, valid_behaviors_file
    )

evaluator = MetricEvaluator(
    labels=group_labels,
    predictions=group_preds,
    metric_functions=[
        AucScore(),
        MrrScore(),
        NdcgScore(k=5),
        NdcgScore(k=10),
    ],
)
evaluator.evaluate()
print("New eval:", evaluator.evaluations)

results_path = Path("results") / "mind_nrms_metrics.json"
results_path.write_text(json.dumps(evaluator.evaluations, indent=2))


model_path = os.path.join(data_path, "model")
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "nrms_ckpt"))

# %% [markdown]
# ## Reference
# \[1\] Wu et al. "Neural News Recommendation with Multi-Head Self-Attention." in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)<br>
# \[2\] Wu, Fangzhao, et al. "MIND: A Large-scale Dataset for News Recommendation" Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. https://msnews.github.io/competition.html <br>
# \[3\] GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/
