"""
TODO:
- Set up LLM as classifier
"""

# Base
from os.path import join, dirname
from copy import deepcopy

# Core
import pandas as pd
from sklearn.base import clone

# Models / sklearn stuff
from imblearn.base import BaseSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# HuggingFace stuff
from sentence_transformers import SentenceTransformer

# Own
from mlresearch.model_selection import ModelSearchCV
from mlresearch.utils import check_pipelines

DATA_PATH = join(dirname(__file__), "data")
RESULTS_PATH = join(dirname(__file__), "results")
ANALYSIS_PATH = join(dirname(__file__), "analysis")


class TrainDataFilter(BaseSampler):
    def __init__(self, labels=["harmful", "benign", "ifeval"]):
        self.labels = labels

    def fit(self, X, y):
        return self

    def resample(self, X, y):
        mask = X["source"].isin(self.labels)
        X_ = X[mask]
        y_ = y[mask]
        return X_, y_

    def fit_resample(self, X, y):
        return self.resample(X, y)

    def _fit_resample(self, X, y):
        pass


CONFIG = {
    "TRAINSET": [
        ("HBI", TrainDataFilter(["harmful", "benign", "ifeval"]), {}),
        ("HI", TrainDataFilter(["harmful", "ifeval"]), {}),
    ],
    # NOTE: DROPCOL WILL RETURN ONLY THE EMBEDDINGS OF RESPONSES
    #       SOURCE WILL BE DROPPED
    "DROPCOL": [
        (
            "DROP",
            ColumnTransformer([("drop", "drop", "source")], remainder="passthrough"),
            {},
        )
    ],
    "CLASSIFIERS": [
        ("CONSTANT", DummyClassifier(strategy="prior"), {}),
        (
            "LR",
            LogisticRegression(max_iter=10000),
            {"penalty": ["l1", "l2"], "solver": ["saga"], "C": [0.1, 1.0]},
        ),
        (
            "KNN",
            KNeighborsClassifier(),
            {
                "n_neighbors": [1, 5, 10],
                "metric": ["euclidean", "cosine"],
            },
        ),
        (
            "MLP",
            MLPClassifier(max_iter=10000),
            {
                "hidden_layer_sizes": [(100,), (10, 10), (50, 50), (100, 100)],
                "alpha": [0.0001, 0.001, 0.01],
            },
        ),
        (
            "XGB",
            XGBClassifier(),
            {
                "n_estimators": [10, 100, 1000],
                "max_depth": [5, 10],
            },
        ),
    ],
    "SCORING": ["precision", "recall", "f1", "accuracy"],
    "N_SPLITS": 10,
    "N_RUNS": 3,
    "RANDOM_STATE": 42,
    "N_JOBS": -1,
}


def get_response_embeddings(df):
    df = df.copy()

    # Load SBERT model
    embedder = SentenceTransformer(
        model_name_or_path="all-MiniLM-L6-v2", similarity_fn_name="cosine"
    )
    df_ = pd.DataFrame(embedder.encode(df["response"].tolist()))
    df_.columns = df_.columns.astype(str)
    df_["response_type"] = df["response_type"]
    df = df_
    return df


def refit_optimal_params(X, y, pipelines, experiment):
    res = deepcopy(experiment.cv_results_)
    df_res = pd.DataFrame(res)
    columns = ["param_est_name", "params", "mean_test_f1"]
    opt_params = (
        df_res[columns]
        .groupby("param_est_name")
        .apply(
            lambda data: data.iloc[data["mean_test_f1"].argmax()], include_groups=False
        )["params"]
    )
    pipelines_optimal = {}
    for param in opt_params:
        est_name = param.pop("est_name")
        pipeline = clone(dict(pipelines)[est_name])
        param = {k.replace(f"{est_name}__", ""): v for k, v in param.items()}
        pipelines_optimal[est_name] = pipeline.set_params(**param).fit(X, y)

    return pipelines_optimal


if __name__ == "__main__":
    df = pd.read_csv(
        join(DATA_PATH, "train_dataset_A_B_llama2024_11_22_17_15_01_280007.csv")
    )
    df_oos = pd.read_csv(join(DATA_PATH, "test_data.csv"))

    df = get_response_embeddings(df)
    df_oos = get_response_embeddings(df_oos)

    df["source"] = (
        ["harmful" for _ in range(1300)]
        + ["benign" for _ in range(1300)]
        + ["ifeval" for _ in range(1300)]
    )

    # DATASET B: WITH BENIGN PROMPTS/OUTPUTS
    X = df.drop(columns="response_type")
    y = df["response_type"].astype(int).values

    pipelines, params = check_pipelines(
        CONFIG["TRAINSET"],
        CONFIG["DROPCOL"],
        CONFIG["CLASSIFIERS"],
        random_state=CONFIG["RANDOM_STATE"],
        n_runs=CONFIG["N_RUNS"],
    )

    experiment_b = ModelSearchCV(
        estimators=pipelines,
        param_grids=params,
        scoring=CONFIG["SCORING"],
        n_jobs=CONFIG["N_JOBS"],
        cv=StratifiedKFold(
            n_splits=CONFIG["N_SPLITS"],
            shuffle=True,
            random_state=CONFIG["RANDOM_STATE"],
        ),
        verbose=2,
        return_train_score=True,
        refit=False,
    ).fit(X, y)

    # Save results
    filename = join(RESULTS_PATH, "param_tuning_results_dataset_B.pkl")
    pd.DataFrame(experiment_b.cv_results_).to_pickle(filename)
    pipelines_b = refit_optimal_params(X, y, pipelines, experiment_b)

    # DATASET A: WITHOUT BENIGN PROMPTS/OUTPUTS
    df = df[df["source"] != "benign"]
    X = df.drop(columns="response_type")
    y = df["response_type"].astype(int).values

    pipelines, params = check_pipelines(
        CONFIG["DROPCOL"],
        CONFIG["CLASSIFIERS"],
        random_state=CONFIG["RANDOM_STATE"],
        n_runs=CONFIG["N_RUNS"],
    )

    experiment_a = ModelSearchCV(
        estimators=pipelines,
        param_grids=params,
        scoring=CONFIG["SCORING"],
        n_jobs=CONFIG["N_JOBS"],
        cv=StratifiedKFold(
            n_splits=CONFIG["N_SPLITS"],
            shuffle=True,
            random_state=CONFIG["RANDOM_STATE"],
        ),
        verbose=2,
        return_train_score=True,
        refit=False,
    ).fit(X, y)

    # Save results
    filename = join(RESULTS_PATH, "param_tuning_results_dataset_A.pkl")
    pd.DataFrame(experiment_a.cv_results_).to_pickle(filename)
    # pipelines_a = refit_optimal_params(X, y, pipelines, experiment_a)

    # OUT OF SAMPLE TESTING
    X_oos = df_oos.drop(columns="response_type")
    y_oos = df_oos["response_type"].astype(int).values

    results_oos = {"target": y_oos}
    for est_name in pipelines_b.keys():
        results_oos[est_name] = pipelines_b[est_name].predict_proba(X_oos)[:, 1]

    # Save results
    filename = join(RESULTS_PATH, "out_of_sample_results.pkl")
    pd.DataFrame(results_oos).to_pickle(filename)
