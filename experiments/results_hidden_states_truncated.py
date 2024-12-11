# Base
from os.path import join, isfile, pardir, dirname
import pickle

# Core
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch

# Models / sklearn stuff
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import GroupKFold

# Own
from mlresearch.model_selection import ModelSearchCV
from mlresearch.utils import check_pipelines

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.append(PROJ_PATH)

# Experiments / model wrappers
from experiments.results_sbert import (
    DATA_PATH,
    RESULTS_PATH,
    CONFIG,
    refit_optimal_params,
    remove_jailbreak_target,
)
from experiments.results_hidden_states import (
    get_hidden_states,
    get_generation_scores,
)
from ctg.new_ctg import ModelWrapper


if __name__ == "__main__":

    # Load LLM (llama)
    cache_dir = "/scratch/jpm9748/"
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir, use_safetensors=True
    )

    try:
        print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))
        model.cuda()  # .to(model.device)
        print("DEBUG::Model succesfully moved to Cuda.")
        print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))
        CUDA = True
    except:  # noqa
        CUDA = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, use_safetensors=True
    )
    m = ModelWrapper(model, tokenizer, mode="topk", k=100, temperature=0.3, cuda=CUDA)

    df = pd.read_csv(
        join(DATA_PATH, "train_dataset_A_B_llama2024_11_22_17_15_01_280007.csv")
    )
    df["response"] = df.apply(remove_jailbreak_target, axis=1)
    prompts = df["prompt"].values
    print("Getting Dataset's A and B hidden states...")
    filename = join(RESULTS_PATH, "hidden_states_truncated.pkl")
    if not isfile(filename):
        df = get_hidden_states(df, m)
        df.to_pickle(filename)
    else:
        df = pd.read_pickle(filename)

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

    filename = join(
        RESULTS_PATH, "param_tuning_results_dataset_B_hidden_states_truncated.pkl"
    )
    if not isfile(filename):
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
        pd.DataFrame(experiment_b.cv_results_).to_pickle(filename)

    experiment_b = pd.read_pickle(filename)
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

    filename = join(
        RESULTS_PATH, "param_tuning_results_dataset_A_hidden_states_truncated.pkl"
    )
    if not isfile(filename):
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
        pd.DataFrame(experiment_a.cv_results_).to_pickle(filename)

    # OUT OF SAMPLE TESTING - MODELS TESTED ON DATASET B ONLY
    llm_name = "llama"
    df_oos = pd.read_csv(join(DATA_PATH, f"oos_test_dataset_{llm_name}_2024_11_25.csv"))
    df_oos["response"] = df_oos.apply(remove_jailbreak_target, axis=1)

    print("Getting OOS hidden states...")
    filename = join(RESULTS_PATH, "hidden_states_truncated_oos.pkl")
    if not isfile(filename):
        df_oos = get_hidden_states(df_oos, m)
        df_oos.to_pickle(filename)
    else:
        df_oos = pd.read_pickle(filename)

    X_oos = df_oos.drop(columns="response_type")
    y_oos = df_oos["response_type"].astype(int).values

    results_oos = {"target": y_oos}
    for est_name in pipelines_b.keys():
        results_oos[est_name] = pipelines_b[est_name].predict_proba(X_oos)[:, 1]

    # Save results
    filename = join(
        RESULTS_PATH, f"out_of_sample_results_{llm_name}_hidden_state_truncated.pkl"
    )
    pd.DataFrame(results_oos).to_pickle(filename)

    # Save classifiers
    for est_name in pipelines_b.keys():
        pickle.dump(
            pipelines_b[est_name],
            open(
                join(
                    RESULTS_PATH,
                    f"MODEL_{est_name.replace('|', '_')}_hidden_states_truncated.pkl",
                ),
                "wb",
            ),
        )

    # Get scores over 100 benefic, 100 dangerous prompts over "time"
    # Using MLP, for Llama outputs
    print("Generating scores over 'time'")
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    df_oos = pd.read_csv(join(DATA_PATH, "oos_test_dataset_llama_2024_11_25.csv"))
    df_oos = (
        df_oos.groupby(["prompt"])
        .sample(1, random_state=CONFIG["RANDOM_STATE"])
        .groupby(["response_type"])
        .sample(100, random_state=CONFIG["RANDOM_STATE"])
    )
    clf = pipelines_b["HBI|DROP|MLP"].steps[-1][-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_safetensors=True)

    response_score_data = [
        get_generation_scores(prompt, response, model, clf)
        for prompt, response in tqdm(zip(df_oos["prompt"], df_oos["response"]))
    ]
    max_len_tokens = max([len(resp_scores) for resp_scores in response_score_data])
    response_score_data = [
        resp_scores + [np.nan for _ in range(max_len_tokens - len(resp_scores))]
        for resp_scores in response_score_data
    ]
    df_oos_time = pd.DataFrame(response_score_data)
    df_oos_time["response"] = df_oos["response"].tolist()
    df_oos_time["response_type"] = df_oos["response_type"].tolist()
    filename = join(RESULTS_PATH, "oos_time_scores_hidden_states_truncated.pkl")
    df_oos_time.to_pickle(filename)
