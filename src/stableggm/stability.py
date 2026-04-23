from __future__ import annotations

import numpy as np
import pandas as pd
import os
from .subsample import run_subsample_pcor
from .channel_pipeline import run_single_channel_pipeline
from .plotting import plot_mcl_inflation_sensitivity, plot_edges_venn

def _validate_expr_df(expr_df: pd.DataFrame) -> pd.DataFrame:
    """
    检查输入表达矩阵是否合法。
    """
    if not isinstance(expr_df, pd.DataFrame):
        raise TypeError("expr_df must be a pandas DataFrame.")
    if expr_df.empty:
        raise ValueError("expr_df is empty.")
    return expr_df

def _validate_intersection_params(
    n_channels: int,
    intersection_mode: str,
    min_presence: int | None
) -> int:
    """
    检查 stability 交集参数，并返回最终 min_presence。
    """
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1")
    if intersection_mode not in {"strict", "soft"}:
        raise ValueError("intersection_mode must be 'strict' or 'soft'")
    if intersection_mode == "strict":
        return n_channels
    if min_presence is None:
        raise ValueError("min_presence must be provided when intersection_mode='soft'")
    if min_presence < 1 or min_presence > n_channels:
        raise ValueError("min_presence must be between 1 and n_channels")
    return min_presence

def _standardize_final_edges(final_edges: pd.DataFrame) -> pd.DataFrame:
    """
    标准化单个 channel 的最终边表。
    至少要求：
    - gene1
    - gene2

    其余列如 weight / score / qvalue 等保留原样。
    """
    if not isinstance(final_edges, pd.DataFrame):
        raise TypeError("final_edges must be a pandas DataFrame.")
    if final_edges.empty:
        return pd.DataFrame(columns=["gene1", "gene2"])
    required_cols = {"gene1", "gene2"}
    missing = required_cols - set(final_edges.columns)
    if missing:
        raise ValueError(f"final_edges missing required columns: {missing}")
    df = final_edges.copy()
    df["gene1"] = df["gene1"].astype(str)
    df["gene2"] = df["gene2"].astype(str)
    return df

def run_single_stability_channel(
    expr_df: pd.DataFrame,
    bacteria: str = "unknown",
    subset_size: int | None = None,
    n_iterations: int | None = None,
    random_state: int | None = None,
    # subsample 自动策略参数
    iteration_cap: int = 2500,
    iteration_trigger: int = 3000,
    max_multiplier: float = 2.0,
    # 调试开关
    store_pcor_matrices: bool = False,
    store_edge_lists: bool = False,
    store_sampled_genes: bool = False,
    # channel pipeline 参数
    method: str = "python_genenet_like",
    output_dir: str | None = None,
    save_intermediate: bool = False,
    make_plots: bool = False,
    fdr_alpha: float = 0.1,
    prob_threshold: float = 0.9,
    cutoff_ggm: float = 0.9,
    plot_r: bool = False,
    plot_inflation_sensitivity: bool = False,
    inflation: float = 2.0,
    run_inflation_scan: bool = False,
    inflation_values: list[float] | None = None,
    negative_weight_policy: str = "abs",
) -> dict:
    """
    单个 stability channel 的完整流程：
    expr_df
      -> run_subsample_pcor()   # 在线聚合，直接得到 aggregated_edge_df
      -> run_single_channel_pipeline()
      -> final edges
    返回
    ----
    dict:
        - aggregated_edges
        - normalized_edge_df
        - screened_matrix
        - final_edges
        - subset_size
        - n_iterations
        - auto_plan
        - method
        - pcor_matrices (可选)
        - edge_lists (可选)
        - sampled_genes (可选)
    """
    expr_df = _validate_expr_df(expr_df)

    # 1. 多轮子采样 + 在线聚合
    subsample_results = run_subsample_pcor(
        expr_df=expr_df,
        subset_size=subset_size,
        n_iterations=n_iterations,
        random_state=random_state,
        iteration_cap=iteration_cap,
        iteration_trigger=iteration_trigger,
        max_multiplier=max_multiplier,
        store_pcor_matrices=store_pcor_matrices,
        store_edge_lists=store_edge_lists,
        store_sampled_genes=store_sampled_genes,
    )
    aggregated_edges = subsample_results["aggregated_edge_df"]
    # 2. 单通道完整流程：normalization -> screen -> edge_selection
    channel_result = run_single_channel_pipeline(
        aggregated_edge_df=aggregated_edges,
        bacteria=bacteria,
        method=method,
        output_dir=output_dir,
        save_intermediate=save_intermediate,
        make_plots=make_plots,
        fdr_alpha=fdr_alpha,
        prob_threshold=prob_threshold,
        cutoff_ggm=cutoff_ggm,
        plot_r=plot_r,
        inflation=inflation,
        run_inflation_scan=run_inflation_scan,
        inflation_values=inflation_values,
        negative_weight_policy=negative_weight_policy,
    )
    # 2.1 可选：绘制 MCL inflation sensitivity 图
    if plot_inflation_sensitivity:
        inflation_df = channel_result.get("inflation_sensitivity_df", None)
        if inflation_df is not None and not inflation_df.empty:
            inflation_plot_path = None
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                inflation_plot_path = os.path.join(
                    output_dir,
                    "mcl_inflation_sensitivity.png"
                )

            plot_mcl_inflation_sensitivity(
                result_df=inflation_df,
                inflation_col="inflation",
                module_count_col="n_modules",
                title=f"MCL Inflation Sensitivity ({bacteria})",
                save_path=inflation_plot_path,
                show=make_plots
            )
    final_edges = _standardize_final_edges(channel_result["final_edges"])
    result = {
        "aggregated_edges": aggregated_edges,
        "normalized_edge_df": channel_result["normalized_edge_df"],
        "screened_matrix": channel_result["screened_matrix"],
        "final_edges": final_edges,
        "subset_size": subsample_results["subset_size"],
        "n_iterations": subsample_results["n_iterations"],
        "auto_plan": subsample_results["auto_plan"],
        "method": method,
    }
    if "inflation_sensitivity_df" in channel_result:
        result["inflation_sensitivity_df"] = channel_result["inflation_sensitivity_df"];

    # 仅调试时附带中间结果
    if "pcor_matrices" in subsample_results:
        result["pcor_matrices"] = subsample_results["pcor_matrices"]
    if "edge_lists" in subsample_results:
        result["edge_lists"] = subsample_results["edge_lists"]
    if "sampled_genes" in subsample_results:
        result["sampled_genes"] = subsample_results["sampled_genes"]
    return result

def run_stability_selection(
    expr_df: pd.DataFrame,
    bacteria: str = "unknown",
    n_channels: int = 5,
    subset_size: int | None = None,
    n_iterations: int | None = None,
    random_state: int | None = None,
    # subsample 自动策略参数
    iteration_cap: int = 2500,
    iteration_trigger: int = 3000,
    max_multiplier: float = 2.0,
    # 调试开关
    store_pcor_matrices: bool = False,
    store_edge_lists: bool = False,
    store_sampled_genes: bool = False,
    # channel pipeline 参数
    method: str = "python_genenet_like",
    output_dir: str | None = None,
    save_intermediate: bool = False,
    make_plots: bool = False,
    fdr_alpha: float = 0.1,
    cutoff_ggm: float = 0.9,
    plot_r: bool = False,
    # stability 参数
    intersection_mode: str = "soft",
    prob_threshold: float = 0.9,
    min_presence: int | None = 2,
    plot_inflation_sensitivity: bool = False,
    plot_venn: bool = False,
    inflation: float = 2.0,
    run_inflation_scan: bool = False,
    inflation_values: list[float] | None = None,
    negative_weight_policy: str = "abs",
) -> dict:
    """
    多个 channel 的稳定性筛选。
    流程：
        对每个 channel：
            subsample(在线聚合) -> channel_pipeline -> final_edges
        再统计每条边在多少个 channel 中出现
        根据 strict / soft 规则保留稳定边
    返回
    ----
    dict:
        - channel_results
        - stability_table
        - stable_edges
        - n_channels
        - min_presence
        - method
    """
    expr_df = _validate_expr_df(expr_df)
    min_presence = _validate_intersection_params(
        n_channels=n_channels,
        intersection_mode=intersection_mode,
        min_presence=min_presence
    )
    rng = np.random.default_rng(random_state)
    channel_results = []
    channel_edge_tables = []

    for channel_idx in range(n_channels):
        channel_seed = int(rng.integers(0, 1_000_000_000))

        channel_output_dir = None
        if output_dir is not None:
            channel_output_dir = f"{output_dir}/channel_{channel_idx + 1}"
        channel_result = run_single_stability_channel(
            expr_df=expr_df,
            bacteria=f"{bacteria}_ch{channel_idx + 1}",
            subset_size=subset_size,
            n_iterations=n_iterations,
            random_state=channel_seed,
            iteration_cap=iteration_cap,
            iteration_trigger=iteration_trigger,
            max_multiplier=max_multiplier,
            store_pcor_matrices=store_pcor_matrices,
            store_edge_lists=store_edge_lists,
            store_sampled_genes=store_sampled_genes,
            method=method,
            output_dir=channel_output_dir,
            save_intermediate=save_intermediate,
            make_plots=make_plots,
            fdr_alpha=fdr_alpha,
            prob_threshold=prob_threshold,
            cutoff_ggm=cutoff_ggm,
            plot_r=plot_r,
            plot_inflation_sensitivity=plot_inflation_sensitivity,
        )
        final_edges = channel_result["final_edges"].copy()
        if not final_edges.empty:
            final_edges["channel"] = channel_idx + 1

        channel_results.append(channel_result)
        channel_edge_tables.append(final_edges)
    # print("plot_venn:",plot_venn)
    if plot_venn:
        edge_sets = {}
        labels = {}
        for i, edge_df in enumerate(channel_edge_tables):
            channel_name = f"channel_{i + 1}"
            edge_sets[channel_name] = set(
                zip(edge_df["gene1"].astype(str), edge_df["gene2"].astype(str))
            ) if not edge_df.empty else set()
            labels[channel_name] = f"Ch{i + 1}"

        venn_save_path = None
        if output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)
            venn_save_path = os.path.join(output_dir, "channel_edges_venn.png")
            # print("edge_sets:",edge_sets)
        plot_edges_venn(
            edge_sets=edge_sets,
            labels=labels,
            title=f"Final Edge Overlap across Channels ({bacteria})",
            save_path=venn_save_path,
            show=make_plots
        )
    non_empty_tables = [df for df in channel_edge_tables if not df.empty]
    # 所有 channel 都没有最终边
    if len(non_empty_tables) == 0:
        stability_table = pd.DataFrame(
            columns=[
                "gene1", "gene2",
                "presence_count", "presence_ratio",
                "weight_mean", "weight_median",
                "score_mean", "qvalue_median"
            ]
        )
        stable_edges = pd.DataFrame(
            columns=[
                "gene1", "gene2",
                "presence_count", "presence_ratio",
                "weight", "score_mean", "qvalue_median"
            ]
        )
        return {
            "channel_results": channel_results,
            "stability_table": stability_table,
            "stable_edges": stable_edges,
            "n_channels": n_channels,
            "min_presence": min_presence,
            "method": method,
        }
    merged_channels = pd.concat(non_empty_tables, ignore_index=True)
    grouped = merged_channels.groupby(["gene1", "gene2"], sort=True)
    rows = []
    for (gene1, gene2), group in grouped:
        presence_count = int(group["channel"].nunique())
        presence_ratio = presence_count / n_channels
        row = {
            "gene1": gene1,
            "gene2": gene2,
            "presence_count": presence_count,
            "presence_ratio": presence_ratio,
        }
        if "weight" in group.columns:
            row["weight_mean"] = float(group["weight"].mean())
            row["weight_median"] = float(group["weight"].median())
        if "score" in group.columns:
            row["score_mean"] = float(group["score"].mean())
        if "qvalue" in group.columns:
            row["qvalue_median"] = float(group["qvalue"].median())
        rows.append(row)
    stability_table = pd.DataFrame(rows)
    sort_cols = ["presence_count"]
    ascending = [False]
    if "score_mean" in stability_table.columns:
        sort_cols.append("score_mean")
        ascending.append(False)
    sort_cols.extend(["gene1", "gene2"])
    ascending.extend([True, True])

    stability_table = stability_table.sort_values(
        by=sort_cols,
        ascending=ascending
    ).reset_index(drop=True)
    # 保留稳定边
    stable_edges = stability_table.loc[
        stability_table["presence_count"] >= min_presence
    ].copy()
    # 定义最终权重
    if "weight_median" in stable_edges.columns:
        stable_edges["weight"] = stable_edges["weight_median"]
    elif "weight_mean" in stable_edges.columns:
        stable_edges["weight"] = stable_edges["weight_mean"]
    keep_cols = ["gene1", "gene2", "presence_count", "presence_ratio"]
    if "weight" in stable_edges.columns:
        keep_cols.append("weight")
    if "score_mean" in stable_edges.columns:
        keep_cols.append("score_mean")
    if "qvalue_median" in stable_edges.columns:
        keep_cols.append("qvalue_median")
    stable_edges = stable_edges[keep_cols].reset_index(drop=True)
    return {
        "channel_results": channel_results,
        "stability_table": stability_table,
        "stable_edges": stable_edges,
        "n_channels": n_channels,
        "min_presence": min_presence,
        "method": method,
    }

def extract_stable_edge_keys(edge_df: pd.DataFrame) -> set:
    """
    提取稳定边键集合 {(gene1, gene2), ...}
    """
    if not isinstance(edge_df, pd.DataFrame):
        raise TypeError("edge_df must be a pandas DataFrame.")

    if edge_df.empty:
        return set()

    required_cols = {"gene1", "gene2"}
    missing = required_cols - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge_df missing required columns: {missing}")

    return set(zip(edge_df["gene1"], edge_df["gene2"]))
