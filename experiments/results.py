"""
TODO:
- Set up LLM as classifier
"""

# Base
from os.path import join, dirname

# Core
import pandas as pd

# Models / sklearn stuff
from sklearn.model_selection import StratifiedKFold
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

CONFIG = {
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
                "hidden_layer_sizes": [(100,), (10, 10), (50, 50)],
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
    "N_SPLITS": 5,
    "N_RUNS": 3,
    "RANDOM_STATE": 42,
    "N_JOBS": -1,
}

if __name__ == "__main__":
    df = pd.read_csv(join(DATA_PATH, "test_data.csv"))

    pipelines, params = check_pipelines(
        CONFIG["CLASSIFIERS"],
        random_state=CONFIG["RANDOM_STATE"],
        n_runs=CONFIG["N_RUNS"],
    )

    # Produce SBERT output embeddings
    embedder = SentenceTransformer(
        model_name_or_path="all-MiniLM-L6-v2", similarity_fn_name="cosine"
    )
    X = embedder.encode(df["response"])
    y = df["response_type"].astype(int).values

    experiment = ModelSearchCV(
        estimators=pipelines,
        param_grids=params,
        scoring=CONFIG["SCORING"],
        n_jobs=CONFIG["N_JOBS"],
        cv=StratifiedKFold(
            n_splits=CONFIG["N_SPLITS"],
            shuffle=True,
            random_state=CONFIG["RANDOM_STATE"],
        ),
        verbose=1,
        return_train_score=True,
        refit=False,
    ).fit(X, y)

    # Save results
    filename = join(RESULTS_PATH, "param_tuning_results.pkl")
    pd.DataFrame(experiment.cv_results_).to_pickle(filename)
