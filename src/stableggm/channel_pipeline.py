from __future__ import annotations

import os
import pandas as pd
from .normalization import normalize_pcor_column
from .edge_selection import select_edges
from .plotting import plot_normalization_distributions
from .clustering import run_mcl_clustering
from .network import build_graph_from_edge_df
import numpy as np
def _validate_aggregated_edge_df(aggregated_edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    检查聚合后的 edge table 是否包含必要列。
    """
    if not isinstance(aggregated_edge_df, pd.DataFrame):
        raise TypeError("aggregated_edge_df must be a pandas DataFrame.")
    required_cols = {"gene1", "gene2", "pcor"}
    missing = required_cols - set(aggregated_edge_df.columns)
    if missing:
        raise ValueError(
            f"aggregated_edge_df is missing required columns: {missing}"
        )
    if aggregated_edge_df.empty:
        raise ValueError("aggregated_edge_df is empty.")
    df = aggregated_edge_df.copy()
    df["gene1"] = df["gene1"].astype(str)
    df["gene2"] = df["gene2"].astype(str)
    df["pcor"] = df["pcor"].astype(float)
    return df

def screen_df(
    edge_df: pd.DataFrame,
    value_col: str = "norm_pcor",
    fill_diagonal: float = 1.0,
    fill_missing: float = 0.0,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> pd.DataFrame:
    """
    将边表转换为对称矩阵。
    参数
    ----
    edge_df : pd.DataFrame
        必须包含 gene1, gene2, value_col
    value_col : str
        作为矩阵值的列名，默认 norm_pcor
    fill_diagonal : float
        对角线填充值，默认 1.0
    fill_missing : float
        缺失边填充值，默认 0.0
    clip_min, clip_max : float
        数值裁剪范围，默认 [-1, 1]
    返回
    ----
    matrix_df : pd.DataFrame
        对称矩阵
    """
    required_cols = {"gene1", "gene2", value_col}
    missing = required_cols - set(edge_df.columns)  
    if missing:
        raise ValueError(f"edge_df is missing required columns: {missing}")
    if edge_df.empty:
        return pd.DataFrame()

    nodes = sorted(set(edge_df["gene1"]).union(set(edge_df["gene2"])))
    matrix_df = pd.DataFrame(
        fill_missing,
        index=nodes,
        columns=nodes,
        dtype=float
    )
    # 对角线设为 1
    for node in nodes:
        matrix_df.loc[node, node] = fill_diagonal
    for _, row in edge_df.iterrows():
        g1 = row["gene1"]
        g2 = row["gene2"]
        val = float(row[value_col])
        if val > clip_max:
            val = clip_max
        elif val < clip_min:
            val = clip_min
        matrix_df.loc[g1, g2] = val
        matrix_df.loc[g2, g1] = val
    return matrix_df

def run_single_channel_pipeline(
    aggregated_edge_df: pd.DataFrame,
    bacteria: str = "unknown",
    method: str = "python_genenet_like",
    output_dir: str | None = None,
    save_intermediate: bool = False,
    make_plots: bool = False,
    plot_prefix: str | None = None,
    # Python 纯实现参数
    fdr_alpha: float = 0.1,
    prob_threshold: float = 0.9,
    # R / GeneNet 参数
    cutoff_ggm: float = 0.9,
    plot_r: bool = False,
    # clustering 参数
    inflation: float = 2.0,
    negative_weight_policy: str = "abs",
    # inflation 扫描参数
    run_inflation_scan: bool = False,
    inflation_values: list[float] | None = None,
) -> dict:
    """
    单通道流程函数：
    aggregated edge table
        -> normalization (Fisher z + mean-centering)
        -> screen（边表转对称矩阵）
        -> edge selection（Python 或 R/GeneNet）
        -> final edges
        -> MCL clustering（正式固定 inflation）
        -> optional inflation scan（仅测试/诊断时）

    参数
    ----
    aggregated_edge_df : pd.DataFrame
        聚合后的边表，至少包含 gene1, gene2, pcor
    bacteria : str
        数据集/菌名，用于输出命名
    method : str
        'python_bh' 或 'r_genenet' 或其他已支持方法
    output_dir : str or None
        若给定，则保存结果到该目录
    save_intermediate : bool
        是否保存中间结果
    make_plots : bool
        是否绘制 normalization 前后分布图
    plot_prefix : str or None
        图文件名前缀，默认使用 bacteria
    fdr_alpha : float
        纯 Python 模式下的 FDR 阈值
    prob_threshold : float
        纯 Python 模式下的 score 阈值
    cutoff_ggm : float
        R/GeneNet 模式下 extract.network 的 cutoff.ggm
    plot_r : bool
        R/GeneNet 模式下是否绘图
    inflation : float
        正式聚类使用的固定 inflation，默认 2.0
    negative_weight_policy : str
        构图时负权处理方式，如 "abs"
    run_inflation_scan : bool
        是否执行 inflation 扫描（测试/诊断时打开）
    inflation_values : list[float] | None
        inflation 扫描列表；若为 None，默认 [1.4, 1.6, 1.8, 2.0, 2.2]

    返回
    ----
    result : dict
        包括：
        - normalized_edge_df
        - screened_matrix
        - final_edges
        - clustering_result
        - inflation_sensitivity_df
        - method
    """
    aggregated_edge_df = _validate_aggregated_edge_df(aggregated_edge_df)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if plot_prefix is None:
        plot_prefix = bacteria

    if inflation_values is None:
        inflation_values = [1.4, 1.6, 1.8, 2.0, 2.2]

    # =========================
    # Step 1. normalization
    # =========================
    normalized_edge_df = normalize_pcor_column(
        aggregated_edge_df,
        pcor_col="pcor"
    )
    if make_plots and output_dir is not None:
        hist_path = os.path.join(output_dir, f"{plot_prefix}_norm_hist.png")
        density_path = os.path.join(output_dir, f"{plot_prefix}_norm_density.png")

        plot_normalization_distributions(
            normalized_edge_df,
            pcor_col="pcor",
            norm_col="norm_pcor",
            hist_path=hist_path,
            density_path=density_path,
            show=False
        )
    # =========================
    # Step 2. screen
    # =========================
    screened_matrix = screen_df(
        normalized_edge_df,
        value_col="norm_pcor"
    )
    # =========================
    # Step 3. edge selection
    # =========================
    final_edges = select_edges(
        screened_matrix,
        method=method,
        fdr_alpha=fdr_alpha,
        prob_threshold=prob_threshold,
        # cutoff_ggm=cutoff_ggm,
        # plot_r=plot_r
    )

    # =========================
    # Step 4. 正式聚类：固定 inflation
    # =========================
    clustering_result = None
    inflation_sensitivity_df = pd.DataFrame(columns=["inflation", "n_modules"])

    if isinstance(final_edges, pd.DataFrame) and not final_edges.empty:
        G = build_graph_from_edge_df(
            final_edges,
            weight_col="weight" if "weight" in final_edges.columns else None,
            negative_weight_policy=negative_weight_policy
        )

        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            # 4.1 正式结果：固定 inflation
            clustering_result = run_mcl_clustering(
                G=G,
                inflation=float(inflation),
                negative_weight_policy=negative_weight_policy
            )

            # 4.2 测试/诊断时：可选扫描
            if run_inflation_scan:
                inflation_rows = []

                for infl in inflation_values:
                    try:
                        scan_result = run_mcl_clustering(
                            G=G,
                            inflation=float(infl),
                            negative_weight_policy=negative_weight_policy
                        )
                        # 兼容不同返回风格
                        if isinstance(scan_result, dict):
                            if "membership_df" in scan_result:
                                membership_df = scan_result["membership_df"]
                                if (
                                    isinstance(membership_df, pd.DataFrame)
                                    and "module_id" in membership_df.columns
                                ):
                                    n_modules = int(membership_df["module_id"].nunique())
                                else:
                                    n_modules = 0
                            elif "clusters" in scan_result:
                                clusters = scan_result["clusters"]
                                n_modules = len(clusters)
                            else:
                                n_modules = 0
                        else:
                            n_modules = 0

                        inflation_rows.append(
                            {
                                "inflation": float(infl),
                                "n_modules": int(n_modules)
                            }
                        )
                    except Exception as e:
                        inflation_rows.append(
                            {
                                "inflation": float(infl),
                                "n_modules": np.nan,
                                "error": str(e)
                            }
                        )

                inflation_sensitivity_df = pd.DataFrame(inflation_rows)

    # =========================
    # Step 5. optional save
    # =========================
    if save_intermediate and output_dir is not None:
        normalized_edge_df.to_csv(
            os.path.join(output_dir, f"after_norm_{bacteria}.csv"),
            index=False
        )
        screened_matrix.to_csv(
            os.path.join(output_dir, f"screen_matrix_{bacteria}.csv")
        )
        final_edges.to_csv(
            os.path.join(output_dir, f"final_edges_{bacteria}.csv"),
            index=False
        )

        if clustering_result is not None and isinstance(clustering_result, dict):
            membership_df = clustering_result.get("membership_df", None)
            if isinstance(membership_df, pd.DataFrame) and not membership_df.empty:
                membership_df.to_csv(
                    os.path.join(output_dir, f"mcl_membership_{bacteria}.csv"),
                    index=False
                )

        if run_inflation_scan and not inflation_sensitivity_df.empty:
            inflation_sensitivity_df.to_csv(
                os.path.join(output_dir, f"inflation_sensitivity_{bacteria}.csv"),
                index=False
            )

    return {
        "normalized_edge_df": normalized_edge_df,
        "screened_matrix": screened_matrix,
        "final_edges": final_edges,
        "clustering_result": clustering_result,
        "inflation_sensitivity_df": inflation_sensitivity_df,
        "method": method,
        "inflation": inflation,
        "run_inflation_scan": run_inflation_scan,
    }
