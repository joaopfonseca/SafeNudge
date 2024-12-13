# from os.path import join, dirname
# import numpy as np
# import pandas as pd
# from ctg.evaluation_main import (
#     test_instruction_following_strict,
#     test_instruction_following_loose
# )
#
# DATA_PATH = join(dirname(__file__), "data")
# RESULTS_PATH = join(dirname(__file__), "results")
# ANALYSIS_PATH = join(dirname(__file__), "analysis")
#
# df_ifeval = pd.read_json(
#     "hf://datasets/google/IFEval/ifeval_input_data.jsonl", lines=True
# ).set_index("prompt")
# df_results = pd.read_csv(join(RESULTS_PATH, "final_results_table.csv")).drop(
#     columns="Unnamed: 0"
# ).set_index("prompt")
#
# df_results = df_results.merge(
#     df_ifeval, left_index=True, right_index=True, how="left"
# ).reset_index()
#
# example_input = df_results.iloc[1]
#
# test_instruction_following_loose(
#     example_input.prompt,
#     example_input.response,
#     example_input.instruction_id_list,
#     example_input.kwargs
# ).follow_all_instructions
#
# ifeval_mask = df_results["dataset"] == "ifeval"
# df_results.loc[ifeval_mask, "ifeval_strict"] = df_results.loc[ifeval_mask].apply(
#     (
#         lambda row: test_instruction_following_strict(
#             row.prompt,
#             row.response,
#             row.instruction_id_list,
#             row.kwargs
#         ).follow_all_instructions
#     ),
#     axis=1
# )
# df_results.loc[ifeval_mask, "ifeval_loose"] = df_results.loc[ifeval_mask].apply(
#     (
#         lambda row: test_instruction_following_loose(
#             row.prompt,
#             row.response,
#             row.instruction_id_list,
#             row.kwargs
#         ).follow_all_instructions
#     ),
#     axis=1
# )
# df_results[["ifeval_strict", "ifeval_loose"]] = df_results[["ifeval_strict", "ifeval_loose"]].astype(float)
