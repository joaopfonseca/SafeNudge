# Base
import os
from os.path import join, dirname, pardir
import pickle
import argparse
import time

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.append(PROJ_PATH)

# Core
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch

# Models / sklearn stuff
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import GroupKFold

# HuggingFace stuff
from evaluate import load

# Own
from mlresearch.model_selection import ModelSearchCV
from mlresearch.utils import check_pipelines

# Experiments / model wrappers
DATA_PATH = "data/"
RESULTS_PATH = "results/"
from ctg.new_ctg import ModelWrapper, CTG
from ctg.perplexity import PerplexityCustom

# model_paths = ['meta-llama/Meta-Llama-3.1-8B-Instruct',
#               'Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2',
#               'meta-llama/Llama-Guard-3-8B']


def load_evaluation_prompts():
    df = pd.read_csv(join(DATA_PATH, "evaluation_prompts.csv"))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="model path for experiment", type=str)
    parser.add_argument("--ctg", help="model path for experiment", action="store_true")
    args = parser.parse_args()

    # Load LLM (llama)
    cache_dir = "/scratch/jpm9748/"
    model_path = args.model_path

    try:
        if torch.cuda.memory_allocated(0) == 0:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, cache_dir=cache_dir, use_safetensors=True
            )

            print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))
            model.cuda()
            print("DEBUG::Model succesfully moved to Cuda.")
            print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))
            CUDA = True
        else:
            print("DEBUG::Model already on Cuda.")
            print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))
    except:  # noqa
        model = AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir=cache_dir, use_safetensors=True
        )

        CUDA = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, use_safetensors=True
    )

    if args.ctg:
        clf = pickle.load(
            open(join(RESULTS_PATH, "MODEL_HBI_DROP_MLP_hidden_states_truncated.pkl"),'rb')
        ).steps[-1][-1]
        m = CTG(model, tokenizer, mode="topk", k=100, temperature=1.0, cuda=CUDA)
    else:
        m = ModelWrapper(
            model, tokenizer, mode="topk", k=100, temperature=1.0, cuda=CUDA
        )

    eval_df = load_evaluation_prompts()
    results_df = pd.DataFrame(
        columns=[
            "dataset",
            "prompt",
            "response",
            "ppl_score",
            "inference_time",
            "num_of_tokens",
            "nudged"
        ]
    )

    for i in range(eval_df.shape[0]):
        # Prompt
        prompt = eval_df.iloc[i].prompt
        # print(prompt)
        # Response
        dataset = eval_df.iloc[i].dataset
        if dataset == "advbench":
            target = eval_df.iloc[i].target
        else:
            target = ""

        start_time = time.time()
        # Response
        if args.ctg:
            # Implement CTG code
            response, _, nudged = m.generate_moderated(
                prompt=prompt, clf=clf, target=target, max_tokens=250, verbose=False
            )
        else:
            response, _ = m.generate(
                prompt=prompt, target=target, max_tokens=250, verbose=False
            )
            nudged = False

        end_time = time.time()
        # Inference time
        inference_time = end_time - start_time

        # Num of tokens
        tokens = tokenizer.batch_decode(
            tokenizer(response)["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        num_of_tokens = len(tokens)

        # PPL
        # perplexity =
        # print(perplexity)
        p = PerplexityCustom()
        score = p.compute(predictions=[prompt], model=model, tokenizer=tokenizer)[
            "perplexities"
        ][0]

        r = {
            "dataset": dataset,
            "prompt": prompt,
            "response": response,
            "ppl_score": score,
            "inference_time": inference_time,
            "num_of_tokens": num_of_tokens,
            "nudged": nudged
        }

        results_df.loc[len(results_df)] = r

    filename = f"{RESULTS_PATH}evaluation_responses_{model_path.replace('/', '-')}_ctg_{args.ctg}.pkl"
    results_df.to_pickle(filename)
