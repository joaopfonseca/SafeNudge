from os.path import join, dirname
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlresearch.latex import make_mean_sem_table, export_longtable
from mlresearch.utils import set_matplotlib_style

DATA_PATH = join(dirname(__file__), "data")
RESULTS_PATH = join(dirname(__file__), "results")
ANALYSIS_PATH = join(dirname(__file__), "analysis")

set_matplotlib_style()


def _format_dataset(row):
    if row["Dataset"] == "A":
        return "A"
    elif row["Dataset"] == "B" and row.name.startswith("HBI"):
        return "B"
    else:
        return "A|B"


def get_mean_sem_results(results):
    results = (
        results
        .groupby("param_est_name")
        .apply(lambda df: df.iloc[df["mean_test_f1"].argmax()], include_groups=False)
    )
    mean_sem = []
    for ms in ["mean_test_", "std_test_"]:
        columns = [
            *results.columns[results.columns.str.startswith(ms)].tolist(),
        ]
        res_ = deepcopy(results)[columns]
        res_.columns = res_.columns.str.replace(ms, "")
        mean_sem.append(res_)
    return mean_sem


def format_all_perf(all_perf):
    all_perf = pd.concat(all_perf)
    all_perf["Dataset"] = all_perf.apply(_format_dataset, axis=1)
    all_perf["Model"] = all_perf.apply(lambda x: x.name.split("|")[-1], axis=1)
    all_perf.reset_index(drop=True, inplace=True)
    all_perf.columns = all_perf.columns.str.title()
    all_perf.sort_values("Dataset", inplace=True)
    all_perf = all_perf[["Dataset", "Model", *all_perf.columns[:-2]]]
    return all_perf


if __name__ == "__main__":
    all_perf = []
    for dataset in ["A", "B"]:
        results = pd.read_pickle(
            join(RESULTS_PATH, f"param_tuning_results_dataset_{dataset}.pkl")
        )
        mean_sem_results = get_mean_sem_results(results)
        perf_table = make_mean_sem_table(
            *mean_sem_results, make_bold=True, decimals=3, axis=0
        )
        perf_table["Dataset"] = dataset
        all_perf.append(perf_table)

    all_perf = format_all_perf(all_perf)
    export_longtable(
        all_perf,
        path=join(ANALYSIS_PATH, "kfold_results.tex"),
        caption="""
        Classifier performance results after parameter tuning over 10-fold cross
        validation.
        """,
        label="tbl:kfold-results",
        index=False
    )

    # Analyze OOS results
    results = pd.read_pickle(
        join(RESULTS_PATH, "out_of_sample_results.pkl")
    )
    target = results.target
    results.drop(columns="target", inplace=True)

    taus = np.linspace(0, 1, num=101)
    ben_pass_perc = {}
    dng_pass_perc = {}
    for tau in taus:
        # Benign
        ben_pass_perc[tau] = (results[target == 0] > tau).mean()

        # Dangerous
        dng_pass_perc[tau] = (results[target == 1] > tau).mean()

    ben_pass_perc = pd.DataFrame(ben_pass_perc).T
    dng_pass_perc = pd.DataFrame(dng_pass_perc).T
    dng_pass_perc.plot.line()
