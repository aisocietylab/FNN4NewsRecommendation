# Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks

Implementation of the paper "Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks" (ECIR 2026, IR-for-Good Track).

<p align="center">
    💻 <a href="https://github.com/aisocietylab/FNN4NewsRecommendation" target="_blank">Code</a> | 🌐 <a href="https://aisocietylab.github.io/FNN4NewsRecommendation/" target="_blank">Website</a> | 📃 <a href="https://arxiv.org/abs/2601.04019" target="_blank">Paper</a> <br>
</p>

![Method Overview](MethodDesign.png)

> **Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks**<br>
> Kevin Innerebner, Stephan Bartl, Markus Reiter-Haas, Elisabeth Lex<br>
> [arXiv:2601.04019](https://arxiv.org/abs/2601.04019)
>
> News recommender systems are increasingly driven by black-box models, offering little transparency for editorial decision-making.
> In this work, we introduce a transparent recommender system that uses fuzzy neural networks to learn human-readable rules from behavioral data for predicting article clicks.
> By extracting the rules at configurable thresholds, we can control rule complexity and thus, the level of interpretability.
> We evaluate our approach on two publicly available news datasets (i.e., MIND and EB-NeRD) and show that we can accurately predict click behavior compared to several established baselines, while learning human-readable rules.
> Furthermore, we show that the learned rules reveal news consumption patterns, enabling editors to align content curation goals with target audience behavior.

[![GitHub stars](https://img.shields.io/github/stars/aisocietylab/FNN4NewsRecommendation?style=social)](https://github.com/aisocietylab/FNN4NewsRecommendation)

## Project Structure

- `configs/`: The model configurations used in the experiments.
- `datasets/`: The datasets used in the experiments. Place the downloaded datasets as described below.
- `results/`: The results of the experiments will be saved here.
- `scripts/`: The scripts used to run the experiments.
- `scripts/baselines/`: The implementations of the baseline models.
- `src/`: The source code for the models.
- `main.py`: The main entry point for training and evaluating the models.
- `pyproject.toml`: The project configuration file.


## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Download the dataset from [here](https://recsys.eb.dk/). Requires you to accept the license. We use `ebnerd_small (80MB)`.
3. Download the MIND dataset from [here](https://msnews.github.io/). Requires you to accept the license. We use **`MIND-small`**.


#### Datasets

Create a `datasets` folder and place the datasets in there. The folder structure should look like this:

```
<project-root>/
├── datasets/
│   ├── ebnerd_small/
│   │   ├── articles.parquet
│   │   ├── train/
│   │   │   ├── behaviors.parquet
│   │   ├── val/
│   │   │   ├── behaviors.parquet
│   ├── mind_small/
│   │   ├── train/
│   │   │   ├── behaviors.tsv
│   │   │   ├── news.tsv
│   │   │   ├── ...
│   │   ├── valid/
│   │   │   ├── behaviors.tsv
│   │   │   ├── news.tsv
│   │   │   ├── ...
```

## Running Experiments

The experiment scripts are located in the `scripts/` directory as bash scripts.

1. `all_experiments.sh`: Runs all experiments (EB-NeRD, MIND, ablation study, sensitivity analysis and baselines).
2. `ebnerd_experiment.sh`: Runs the EB-NeRD experiments.
3. `mind_experiment.sh`: Runs the MIND experiments.
4. `ablation_study.sh`: Runs the ablation study experiments.
5. `sensitivity_analysis.sh`: Runs the sensitivity analysis experiments. Takes multiple hours.
6. `baselines.sh`: Runs the baseline experiments.

# Citation

If you find this work helpful, please cite the paper. The BibTeX for the pre-print is below. Please update it with the proceeding details when published.

```bibtex
@misc{innerebner2026modelingbehavioralpatternsnews,
      title={Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks}, 
      author={Kevin Innerebner and Stephan Bartl and Markus Reiter-Haas and Elisabeth Lex},
      year={2026},
      eprint={2601.04019},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.04019}, 
}
```