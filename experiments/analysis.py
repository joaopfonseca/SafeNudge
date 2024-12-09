from os.path import join, dirname
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import get_scorer
from mlresearch.latex import make_mean_sem_table, export_longtable, make_bold
from mlresearch.utils import set_matplotlib_style

# Retrieve metrics used
from experiments.results import CONFIG

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


def get_oos_performance_score(results, threshold=0.5):
    results = results.copy()
    results = (results > threshold).astype(int)
    target = results.target

    metrics = CONFIG["SCORING"]
    metrics = {m: get_scorer(m)._score_func for m in metrics}
    results = pd.DataFrame(
        {
            m: results.apply(lambda col: metrics[m](col, target))
            for m in metrics.keys()
        }
    )
    results["Dataset"] = "B"
    results["Dataset"] = results.apply(
        lambda row: _format_dataset(row)[0], axis=1
    )
    results["Model"] = results.apply(lambda x: x.name.split("|")[-1], axis=1)
    results.sort_values("Dataset", inplace=True)
    results = results[["Dataset", "Model", *results.columns[:-2]]]
    return results


def make_oos_line_chart(results, true_target=0, ax=None):
    results = results.copy()
    target = results.target
    results.drop(columns="target", inplace=True)

    taus = np.linspace(0, 1, num=101)
    pass_perc = {tau: (results[target == true_target] > tau).mean() for tau in taus}

    pass_perc = pd.DataFrame(pass_perc).T
    ax = pass_perc.plot.line(ax=ax)
    return ax


if __name__ == "__main__":
    # K-fold / Parameter tuning results
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
    # fig, axes = plt.subplots(1, 4, sharey=True, figsize=(6, 2))
    # i = 0
    metrics = {}
    for llm_name in ["llama", "mistral"]:
        results = pd.read_pickle(
            join(RESULTS_PATH, f"out_of_sample_results_{llm_name}.pkl")
        )
        results = results[
            results.columns[results.columns.str.startswith("HBI")].tolist()
            + ["target"]
        ]
        results.columns = results.columns.str.replace("HBI|DROP|", "")
        results.drop(columns="CONSTANT", inplace=True)

        # Line charts for benign and dangerous
        for true_target in range(2):
            target_type = "dangerous" if true_target else "benign"
            make_oos_line_chart(
                results, true_target=true_target  # , ax=axes[i]
            )

            plt.ylabel(r"Rejection (\%)")
            plt.legend(
                labels=results.columns.drop("target"),
                loc="lower center",
                ncol=results.columns.size-1,
                bbox_to_anchor=(0, 0.9, 1, 0.5)
            )
            plt.savefig(
                join(ANALYSIS_PATH, f"linechart_{target_type}_{llm_name}.pdf"),
                format="pdf",
                bbox_inches="tight",
                transparent=True
            )
            # axes[i].legend([])
            # axes[i].set_xlabel(f"{llm_name.title()} / {target_type}")
            # i += 1

        # Get performance scores for given threshold (as a table)
        threshold = 0.5
        oos_performance = get_oos_performance_score(results, threshold=threshold)
        oos_performance = make_bold(
            oos_performance.set_index(["Dataset", "Model"]), axis=0
        )
        export_longtable(
            oos_performance.reset_index(),
            path=join(ANALYSIS_PATH, f"oos_performance_{llm_name}.tex"),
            caption=f"""
            Classifier performance results after parameter tuning over an out-of-sample
            dataset with responses generated by {llm_name.title()}
            ($\\tau = {threshold}$).
            """,
            label=f"tbl:oos-performance-{llm_name}",
            index=False
        )

    # axes[0].set_ylabel(r"Rejection (%)")
    # fig.legend(
    #     labels=results.columns.drop("target"),
    #     loc="lower center",
    #     ncol=results.columns.size-1,
    #     bbox_to_anchor=(0, 0.9, 1, 0.5)
    # )
    # plt.savefig(
    #     join(ANALYSIS_PATH, "linechart_oos_all.pdf"),
    #     format="pdf",
    #     bbox_inches="tight",
    #     transparent=True
    # )

    # Get scores over tokens for G
    df_time_scores = pd.read_pickle(join(RESULTS_PATH, "oos_time_scores.pkl"))
    df_time_scores = (
        df_time_scores.drop(columns="response").melt(["response_type"]).dropna()
    )

    sns.lineplot(
        df_time_scores[df_time_scores["variable"] < 100],
        x="variable",
        y="value",
        hue="response_type"
    )
    plt.savefig(
        join(ANALYSIS_PATH, "linechart_scores_over_time_oos_100_each_type.pdf"),
        format="pdf",
        bbox_inches="tight",
        transparent=True
    )
