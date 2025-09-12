import numpy as np
import pandas as pd
from optuna import create_study, TrialPruned
from optuna.samplers import TPESampler, CmaEsSampler
from sklearn.metrics import (
    accuracy_score,                  # Accuracy
    f1_score,                        # F1
    precision_score,                 # Precision
    recall_score,                    # Recall
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
    median_absolute_error,           # MdAE
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    make_scorer
)

import os
import pickle
import warnings
from typing import Any, Callable

from preprocessing.preprocessing import Dataset

warnings.simplefilter("ignore", Warning)

def optimize_hyperparameters(
        dataset: Dataset,
        model_class: Any,
        param_grid: dict,
        n_trials: int = 100,
        scoring: Callable | str = "accuracy",
        cv: int = 5,
        direction: str = "maximize",
        verbose: bool = True
):
    """High-level wrapper for hyperparameters optimization"""

    if not isinstance(dataset, Dataset):
        raise ValueError(f"Incorrect data format. The `Dataset` class was expected, but got {type(dataset)}.")

    use_cv = False
    if dataset.val is None:
        use_cv = True

    metric_dict = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
        'mae': mean_absolute_error,
        'mape': mean_absolute_percentage_error,
        'mdae': median_absolute_error,
        'mse': mean_squared_error,
        'rmse': root_mean_squared_error,
    }

    if callable(scoring):
        scoring_func = scoring
    elif scoring in metric_dict:
        scoring_func = metric_dict[scoring]
    else:
        raise ValueError(f"Unsupported metrics: {scoring}")





def global_objective(trial):
    # Grid for UMAP and model initialization
    n_neighbors = trial.suggest_int("n_neighbors", 20, 75, step=5)
    spread = trial.suggest_float("spread", 6.0, 10.0, step=0.25)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.4, step=0.05)
    n_components = trial.suggest_int("n_components", 20, 100, step=5)
    neg_rate = trial.suggest_int("neg_rate", 5, 10)
    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        negative_sample_rate=neg_rate,
        metric="euclidean",
        # Details about bug fixing with nnd_graph_degree:
        # 1. https://github.com/rapidsai/cuml/issues/6091
        # 2. https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/
        build_algo="nn_descent",
        build_kwds={"nnd_graph_degree": n_neighbors + 1},
        random_state=RANDOM_STATE
    )
    # Grid for HDBSCAN and model initialization
    min_cluster_size = trial.suggest_int("min_cluster_size", 100, 750, step=50)
    min_samples = trial.suggest_int("min_samples", 100, 700, step=50)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="leaf"
    )

    embeddings_reduced = reducer.fit_transform(embeddings)
    normalized_embeddings_reduced = normalize(embeddings_reduced)
    labels = clusterer.fit_predict(normalized_embeddings_reduced)

    # Mask for non-noise examples
    valid = labels != -1
    cluster_counts = np.unique(labels[valid], return_counts=True)
    n_clusters = len(cluster_counts[0])

    # Moulavi D. et al. Density-based clustering validation //Proceedings of the 2014 SIAM
    # international conference on data mining. – Society for Industrial and Applied
    # Mathematics, 2014. – С. 839-847.
    # varies in range [-1, 1] -> higher is better
    dbcv_index = validity_index(
        normalized_embeddings_reduced.astype("float64"),
        labels,
        metric="euclidean"
    )

    # Calculating important statistics for analysis of under- and over-fitting
    min_cluster, max_cluster = int(cluster_counts[1].min()), int(cluster_counts[1].max())
    noise_frac = round(float(1 - np.sum(valid) / N), 4)
    # Assignment of trial attributes for analysis of under- and over-fitting
    trial.set_user_attr(f"noise_fraction", noise_frac)
    trial.set_user_attr(f"clusters", n_clusters)
    trial.set_user_attr(f"min_cluster", min_cluster)
    trial.set_user_attr(f"max_cluster", max_cluster)

    # Updating the dictionary with better results
    if dbcv_index > best_artifacts["dbcv"]:
        best_artifacts["reducer"] = reducer
        best_artifacts["clusterer"] = clusterer
        best_artifacts["noise_frac"] = noise_frac
        best_artifacts["dbcv"] = dbcv_index
    return dbcv_index