"""
TODO:
- Set up LLM as classifier
"""

# Base
from os.path import join, dirname
from copy import deepcopy
import pickle

# Core
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.base import clone

# Models / sklearn stuff
from imblearn.base import BaseSampler
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# HuggingFace stuff
from transformers import AutoTokenizer
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


def remove_jailbreak_target(row):
    if row["response_type"] == 1.0:
        idx = row["response"].find("\n\n")
        if idx > -1:
            return row["response"][idx + 2 :]
    return row["response"]


def get_response_embeddings(df, embedder):
    df = df.copy()

    df_ = pd.DataFrame(embedder.encode(df["response"].tolist()))
    df_.columns = df_.columns.astype(str)
    df_["response_type"] = df["response_type"]
    df = df_
    return df


def refit_optimal_params(X, y, pipelines, experiment):
    df_res = deepcopy(experiment)
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


def get_generation_scores(response, tokenizer, embedder, clf):
    tokens = tokenizer.batch_decode(
        tokenizer(response)["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    scores = [
        clf.predict_proba([embedder.encode("".join(tokens[:i]))])[0][1]
        for i in range(len(tokens))
    ]
    return scores


if __name__ == "__main__":

    # Load SBERT model
    embedder = SentenceTransformer(
        model_name_or_path="all-MiniLM-L6-v2", similarity_fn_name="cosine"
    )

    df = pd.read_csv(
        join(DATA_PATH, "train_dataset_A_B_llama2024_11_22_17_15_01_280007.csv")
    )
    df["response"] = df.apply(remove_jailbreak_target, axis=1)
    prompts = df["prompt"].values
    df = get_response_embeddings(df, embedder)

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

    group_kfold = GroupKFold(
        n_splits=CONFIG["N_SPLITS"],
    )
    splits = group_kfold.split(X, y, prompts)
    experiment_b = ModelSearchCV(
        estimators=pipelines,
        param_grids=params,
        scoring=CONFIG["SCORING"],
        n_jobs=CONFIG["N_JOBS"],
        cv=splits,
        verbose=2,
        return_train_score=True,
        refit=False,
    ).fit(X, y)

    # Save results
    filename = join(RESULTS_PATH, "param_tuning_results_dataset_B_sbert.pkl")
    experiment_b = pd.DataFrame(experiment_b.cv_results_)
    experiment_b.to_pickle(filename)
    pipelines_b = refit_optimal_params(X, y, pipelines, experiment_b)

    # DATASET A: WITHOUT BENIGN PROMPTS/OUTPUTS
    benign_mask = df["source"] != "benign"
    df = df[benign_mask]
    prompts = prompts[benign_mask]
    X = df.drop(columns="response_type")
    y = df["response_type"].astype(int).values

    pipelines, params = check_pipelines(
        CONFIG["DROPCOL"],
        CONFIG["CLASSIFIERS"],
        random_state=CONFIG["RANDOM_STATE"],
        n_runs=CONFIG["N_RUNS"],
    )

    group_kfold = GroupKFold(
        n_splits=CONFIG["N_SPLITS"],
    )
    splits = group_kfold.split(X, y, prompts)
    experiment_a = ModelSearchCV(
        estimators=pipelines,
        param_grids=params,
        scoring=CONFIG["SCORING"],
        n_jobs=CONFIG["N_JOBS"],
        cv=splits,
        verbose=2,
        return_train_score=True,
        refit=False,
    ).fit(X, y)

    # Save results
    filename = join(RESULTS_PATH, "param_tuning_results_dataset_A_sbert.pkl")
    pd.DataFrame(experiment_a.cv_results_).to_pickle(filename)

    # OUT OF SAMPLE TESTING - MODELS TESTED ON DATASET B ONLY
    for llm_name in ["llama", "mistral"]:
        df_oos = pd.read_csv(
            join(DATA_PATH, f"oos_test_dataset_{llm_name}_2024_11_25.csv")
        )
        df_oos["response"] = df_oos.apply(remove_jailbreak_target, axis=1)
        df_oos = get_response_embeddings(df_oos)

        X_oos = df_oos.drop(columns="response_type")
        y_oos = df_oos["response_type"].astype(int).values

        results_oos = {"target": y_oos}
        for est_name in pipelines_b.keys():
            results_oos[est_name] = pipelines_b[est_name].predict_proba(X_oos)[:, 1]

        # Save results
        filename = join(RESULTS_PATH, f"out_of_sample_results_{llm_name}_sbert.pkl")
        pd.DataFrame(results_oos).to_pickle(filename)

    # Save classifiers
    for est_name in pipelines_b.keys():
        pickle.dump(
            pipelines_b[est_name],
            open(
                join(RESULTS_PATH, f"MODEL_{est_name.replace('|', '_')}_sbert.pkl"),
                "wb",
            ),
        )

    # Get scores over 100 benefic, 100 dangerous prompts over "time"
    # Using MLP, for Llama outputs
    print("Generating scores over 'time'")
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    df_oos = pd.read_csv(join(DATA_PATH, "oos_test_dataset_llama_2024_11_25.csv"))
    df_oos["response"] = df_oos.apply(remove_jailbreak_target, axis=1)
    df_oos = (
        df_oos.groupby(["prompt"])
        .sample(1, random_state=CONFIG["RANDOM_STATE"])
        .groupby(["response_type"])
        .sample(100, random_state=CONFIG["RANDOM_STATE"])
    )
    clf = pipelines_b["HBI|DROP|MLP"].steps[-1][-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_safetensors=True)

    response_score_data = [
        get_generation_scores(response, tokenizer, embedder, clf)
        for response in tqdm(df_oos["response"])
    ]
    max_len_tokens = max([len(resp_scores) for resp_scores in response_score_data])
    response_score_data = [
        resp_scores + [np.nan for _ in range(max_len_tokens - len(resp_scores))]
        for resp_scores in response_score_data
    ]
    df_oos_time = pd.DataFrame(response_score_data)
    df_oos_time["response"] = df_oos["response"].tolist()
    df_oos_time["response_type"] = df_oos["response_type"].tolist()
    filename = join(RESULTS_PATH, "oos_time_scores_sbert.pkl")
    df_oos_time.to_pickle(filename)
