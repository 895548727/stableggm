from __future__ import annotations

import os, re
import pandas as pd
from .preprocess import preprocess_expression
from .stability import run_stability_selection
from typing import Dict, List, Optional
# from matplotlib.patches import Patch
from .network import (
    build_graph_from_edges,
    summarize_graph,
    get_node_table,
    get_edge_table,
)
from .clustering import run_mcl_clustering
from .enrich import enrich_modules
from .plotting import (
    # normalization / batch
    plot_normalization_distributions,
    plot_batch_correction_boxplots,
    plot_batch_correction_pca,
    # network
    draw_graph,
    draw_largest_component,
    draw_graph_by_degree,
    draw_graph_by_module,
    draw_largest_component_by_module,
    draw_module_subgraph,
    plot_module_size_distribution,
    plot_degree_distribution,
    plot_component_size_distribution,
    plot_edge_weight_distribution,
    plot_edge_weight_density,
    plot_top_degree_genes,
    plot_top_weighted_degree_genes,
    # stability
    plot_presence_distribution_combined,
    plot_top_stable_edges,
    # enrichment
    plot_enrichment_bubble,
    plot_degree_distribution_loglog
)

def _extract_runtime_params_from_stability(stability_result: dict) -> dict:
    """
    从 stability_result 中尽量提取真实运行时参数。
    """
    subset_size = None
    n_iterations = None
    auto_plan = None

    channel_results = stability_result.get("channel_results", [])
    if len(channel_results) > 0:
        first_channel = channel_results[0]
        subset_size = first_channel.get("subset_size", None)
        n_iterations = first_channel.get("n_iterations", None)
        auto_plan = first_channel.get("auto_plan", None)

    return {
        "subset_size": subset_size,
        "n_iterations": n_iterations,
        "auto_plan": auto_plan,
    }

def _safe_make_dir(path: str | None):
    if path is not None:
        os.makedirs(path, exist_ok=True)

def _save_module_subgraphs(
    G,
    membership_df,
    output_dir: str,
    bacteria: str,
    top_n_modules: int = 5
):
    """
    保存前 top_n_modules 个模块的子图。
    """
    if membership_df.empty:
        return

    module_sizes = (
        membership_df.groupby("module_id")["gene"]
        .count()
        .sort_values(ascending=False)
    )

    module_ids = list(module_sizes.head(top_n_modules).index)

    for module_id in module_ids:
        draw_module_subgraph(
            G,
            membership_df,
            module_id=module_id,
            with_labels=True,
            title=f"{bacteria} Module {module_id}",
            save_path=os.path.join(
                output_dir,
                f"{bacteria}_module_{module_id}_subgraph.png"
            ),
            show=False
        )

def _run_and_plot_enrichment(
    membership_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    output_dir: str,
    bacteria: str,
    background_genes: Optional[List[str]] = None,
    top_n_modules: int = 10,
    fdr_alpha: float = 0.05,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    对基因模块分别进行 GO 和 KEGG 富集分析，保存结果并绘图。

    参数
    ----
    membership_df : pd.DataFrame
        必须包含两列：'gene', 'module_id'
    annotation_df : pd.DataFrame
        必须包含三列：'gene_id', 'GO_terms', 'KEGG_pathways_en'
        (GO_terms 中多个条目用逗号分隔，KEGG_pathways_en 中多个通路用分号分隔)
    output_dir : str
        输出根目录，将在其中创建 GO_enrichment 和 KEGG_enrichment 子目录
    bacteria : str
        菌株名称，用于输出文件名
    background_genes : list[str] or None
        背景基因集；若为 None，则使用 membership_df 中所有基因
    top_n_modules : int
        仅对基因数最多的前 top_n_modules 个模块绘制气泡图
    fdr_alpha : float
        FDR 阈值，用于富集分析校正

    返回
    ----
    result_dict : dict
        {'GO': {module_id: enrichment_df}, 'KEGG': {module_id: enrichment_df}}
    """
    # ------------------- 1. 参数校验 -------------------
    required_cols = {'gene_id', 'GO_terms_en', 'KEGG_pathways_en'}
    if not required_cols.issubset(annotation_df.columns):
        raise ValueError(f"annotation_df 必须包含列 {required_cols}")
    if not {'gene', 'module_id'}.issubset(membership_df.columns):
        raise ValueError("membership_df 必须包含 'gene' 和 'module_id' 列")

    # ------------------- 2. 构建 GO 两列表格 -------------------
    go_df = annotation_df[['gene_id', 'GO_terms_en']].copy()
    go_df = go_df[go_df['GO_terms_en'].notna() & (go_df['GO_terms_en'] != '-')]
    go_df['GO_terms_en'] = go_df['GO_terms_en'].str.split(';')
    go_df = go_df.explode('GO_terms_en')
    go_df = go_df[go_df['GO_terms_en'].str.strip() != '']
    go_df = go_df.drop_duplicates().reset_index(drop=True)
    go_df.columns = ['gene', 'term']

    # ------------------- 3. 构建 KEGG 两列表格 -------------------
    kegg_df = annotation_df[['gene_id', 'KEGG_pathways_en']].copy()
    kegg_df = kegg_df[kegg_df['KEGG_pathways_en'].notna() & (kegg_df['KEGG_pathways_en'] != '-')]
    kegg_df['KEGG_pathways_en'] = kegg_df['KEGG_pathways_en'].str.split(';')
    kegg_df = kegg_df.explode('KEGG_pathways_en')
    kegg_df['KEGG_pathways_en'] = kegg_df['KEGG_pathways_en'].str.strip()
    filtered_df = kegg_df[kegg_df['KEGG_pathways_en'].str.contains(r'[（(]', na=False, regex=True)]
    filtered_df.columns = ['gene', 'term']

    # ------------------- 4. 准备背景基因集 -------------------
    if background_genes is None:
        background_genes = list(membership_df['gene'].unique())

    # ------------------- 5. 定义内部辅助函数：富集 + 保存 + 绘图 -------------------
    def _process_one_type(
        anno_two_col: pd.DataFrame,
        subdir_name: str,
    ) -> Dict[int, pd.DataFrame]:
        # 创建子目录
        enrich_dir = os.path.join(output_dir, subdir_name, "enrichment")
        os.makedirs(enrich_dir, exist_ok=True)

        # 调用 enrich_modules（需要用户已定义该函数）
        module_results = enrich_modules(
            membership_df=membership_df,
            annotation_df=anno_two_col,
            background_genes=background_genes,
            fdr_alpha=fdr_alpha,
        )

        # 计算模块大小，找出前 top_n_modules 个大模块用于绘图
        module_sizes = membership_df.groupby("module_id")["gene"].count().sort_values(ascending=False)
        top_modules = module_sizes.head(top_n_modules).index.tolist()

        # 保存每个模块的富集结果 CSV 并绘制气泡图
        for module_id, enrich_df in module_results.items():
            csv_path = os.path.join(enrich_dir, f"{bacteria}_module_{module_id}_enrichment.csv")
            enrich_df.to_csv(csv_path, index=False)
            if enrich_df.empty:
                continue
            if module_id in top_modules:
                # 假设 plot_enrichment_bubble 已定义，且能接受 save_path 参数
                plot_enrichment_bubble(
                    enrich_df,
                    top_n=20,
                    save_path=os.path.join(enrich_dir, f"{bacteria}_module_{module_id}_bubble.png"),
                    show=False
                )
        return module_results
    # ------------------- 6. 分别处理 GO 和 KEGG -------------------
    go_results = _process_one_type(go_df, "GO_enrichment")
    kegg_results = _process_one_type(filtered_df, "KEGG_enrichment")
    return {"GO": go_results, "KEGG": kegg_results}

def run_stableggm_pipeline(
    expr_df: pd.DataFrame,
    output_dir: str,
    bacteria: str = "demo",

    # preprocess
    data_type: str = "microarray",
    normalization: str = None,
    min_expression: float | None = None,
    zero_threshold: float = 0.5,
    batch_series=None,
    microarray_logged: bool = False,

    # stability / subsample
    n_channels: int = 3,
    subset_size: int | None = None,
    n_iterations: int | None = None,
    random_state: int = 123,
    iteration_cap: int = 2500,
    iteration_trigger: int = 3000,
    max_multiplier: float = 2.0,
    intersection_mode: str = "soft",
    min_presence: int = 2,
    plot_venn: bool = True,
    plot_inflation_sensitivity: bool = False,
    # edge selection
    method: str = "python_genenet_like",
    fdr_alpha: float = 0.1,
    prob_threshold: float = 0.9,
    score_threshold: float | None = None,
    cutoff_ggm: float = 0.9,

    # clustering
    inflation: float = 1.2,
    run_inflation_scan: bool = False,
    inflation_values: list[float] | None = None,
    negative_weight_policy: str = "abs",

    # enrichment
    annotation_df: pd.DataFrame | None = None,
    enrichment_fdr_alpha: float = 0.05,
    # output / debug
    save_intermediate: bool = True,
    make_plots: bool = True,
    store_pcor_matrices: bool = False,
    store_edge_lists: bool = False,
    store_sampled_genes: bool = False,
):
    """
    完整总入口：
    expression matrix
      -> preprocess
      -> stability selection
      -> stable edges
      -> graph
      -> clustering
      -> optional enrichment
      -> save results + plots
    """
    _safe_make_dir(output_dir)

    # =====================================================
    # Step 0. 保存原始输入
    # =====================================================
    expr_df.to_csv(
        os.path.join(output_dir, f"{bacteria}_input_expression.csv")
    )
    # =====================================================
    # Step 1. preprocess
    # =====================================================
    preprocess_result = preprocess_expression(
        expr_df=expr_df,
        data_type=data_type,
        normalization=normalization,
        min_expression=min_expression,
        batch_series=batch_series,
        zero_threshold=zero_threshold,
        microarray_logged=microarray_logged,
    )
    expr_proc = preprocess_result["expr_preprocessed"]
    expr_before_batch = preprocess_result["expr_before_batch"]
    expr_after_batch = preprocess_result["expr_after_batch"]

    expr_proc.to_csv(
        os.path.join(output_dir, f"{bacteria}_preprocessed_expression.csv")
    )

    # ComBat 前后箱线图（自动绘制）
    if make_plots and batch_series is not None and expr_after_batch is not None:
        plot_batch_correction_boxplots(
            expr_before=expr_before_batch,
            expr_after=expr_after_batch,
            batch_series=batch_series,
            save_path=os.path.join(output_dir, f"{bacteria}_combat_boxplot.png"),
            show=False
        )
        plot_batch_correction_pca(
            expr_before=expr_before_batch,
            expr_after=expr_after_batch,
            batch_series=batch_series,
            save_path=os.path.join(output_dir, f"{bacteria}_combat_pca.png"),
            show=False
        )
    if score_threshold is not None:
        prob_threshold = score_threshold

    # =====================================================
    # Step 2. stability selection
    # =====================================================
    stability_result = run_stability_selection(
        expr_df=expr_proc,
        bacteria=bacteria,
        n_channels=n_channels,
        subset_size=subset_size,
        n_iterations=n_iterations,
        random_state=random_state,
        iteration_cap=iteration_cap,
        iteration_trigger=iteration_trigger,
        max_multiplier=max_multiplier,
        store_pcor_matrices=store_pcor_matrices,
        store_edge_lists=store_edge_lists,
        store_sampled_genes=store_sampled_genes,
        method=method,
        output_dir=os.path.join(output_dir, "channels"),
        save_intermediate=save_intermediate,
        make_plots=False,
        fdr_alpha=fdr_alpha,
        prob_threshold=prob_threshold,
        cutoff_ggm=cutoff_ggm,
        plot_r=False,
        intersection_mode=intersection_mode,
        min_presence=min_presence,
        plot_venn=plot_venn,
        run_inflation_scan=run_inflation_scan,
        inflation_values=inflation_values,
        plot_inflation_sensitivity=plot_inflation_sensitivity
    )
    stability_table = stability_result["stability_table"]
    stable_edges = stability_result["stable_edges"]
    stability_table.to_csv(
        os.path.join(output_dir, f"{bacteria}_stability_table.csv"),
        index=False
    )
    stable_edges.to_csv(
        os.path.join(output_dir, f"{bacteria}_stable_edges.csv"),
        index=False
    )
    runtime_params = _extract_runtime_params_from_stability(stability_result)
    # =====================================================
    # Step 2.5. 标准化前后分布图
    # 用第一个 channel 的 normalized_edge_df 来画
    # =====================================================
    if make_plots:
        channel_results = stability_result.get("channel_results", [])
        if len(channel_results) > 0:
            normalized_edge_df = channel_results[0].get("normalized_edge_df", None)
            if normalized_edge_df is not None and not normalized_edge_df.empty:
                plot_normalization_distributions(
                    normalized_edge_df,
                    pcor_col="pcor",
                    norm_col="norm_pcor",
                    hist_path=os.path.join(output_dir, f"{bacteria}_norm_hist.png"),
                    density_path=os.path.join(output_dir, f"{bacteria}_norm_density.png"),
                    show=False
                )

    # =====================================================
    # Step 3. 如果没有稳定边，提前结束
    # =====================================================
    if stable_edges.empty:
        summary = {
            "bacteria": bacteria,
            "status": "no_stable_edges",
            "n_input_genes": int(expr_df.shape[0]),
            "n_input_samples": int(expr_df.shape[1]),
            "n_preprocessed_genes": int(expr_proc.shape[0]),
            "n_preprocessed_samples": int(expr_proc.shape[1]),
            "n_stable_edges": 0,
            "n_nodes": 0,
            "n_graph_edges": 0,
            "graph_density": 0.0,
            "average_degree": 0.0,
            "average_clustering": 0.0,
            "n_connected_components": 0,
            "largest_component_size": 0,
            "n_modules": 0,
            "largest_module_size": 0,
            "method": method,
            "n_channels": int(n_channels),
            "subset_size": runtime_params["subset_size"],
            "n_iterations": runtime_params["n_iterations"],
            "inflation": float(inflation),
            "intersection_mode": intersection_mode,
            "min_presence": int(stability_result["min_presence"]),
        }

        pd.DataFrame([summary]).to_csv(
            os.path.join(output_dir, f"{bacteria}_pipeline_summary.csv"),
            index=False
        )

        if runtime_params["auto_plan"] is not None:
            pd.DataFrame([runtime_params["auto_plan"]]).to_csv(
                os.path.join(output_dir, f"{bacteria}_auto_plan.csv"),
                index=False
            )

        return {
            "expr_proc": expr_proc,
            "stability_result": stability_result,
            "stable_edges": stable_edges,
            "graph": None,
            "clustering_result": None,
            "module_enrichment_results": {},
            "summary": summary,
        }

    # =====================================================
    # Step 4. network construction
    # =====================================================
    G = build_graph_from_edges(
        stable_edges,
        weight_col="weight"
    )
    graph_summary = summarize_graph(G)
    node_table = get_node_table(G)
    edge_table = get_edge_table(G)

    node_table.to_csv(
        os.path.join(output_dir, f"{bacteria}_node_table.csv"),
        index=False
    )
    edge_table.to_csv(
        os.path.join(output_dir, f"{bacteria}_edge_table_from_graph.csv"),
        index=False
    )
    pd.DataFrame([graph_summary]).to_csv(
        os.path.join(output_dir, f"{bacteria}_graph_summary.csv"),
        index=False
    )

    # =====================================================
    # Step 5. clustering
    # =====================================================
    clustering_result = run_mcl_clustering(
        G,
        inflation=inflation,
        negative_weight_policy=negative_weight_policy,
        include_gene_list_in_summary=False
    )

    membership_df = clustering_result["membership_df"]
    module_summary_df = clustering_result["summary_df"]

    membership_df.to_csv(
        os.path.join(output_dir, f"{bacteria}_module_membership.csv"),
        index=False
    )
    module_summary_df.to_csv(
        os.path.join(output_dir, f"{bacteria}_module_summary.csv"),
        index=False
    )

    # =====================================================
    # Step 6. enrichment（可选）
    # =====================================================
    module_enrichment_results = {}
    if annotation_df is not None and not annotation_df.empty:
        background_genes = list(expr_proc.index)
        module_enrichment_results = _run_and_plot_enrichment(
            membership_df=membership_df,
            annotation_df=annotation_df,
            output_dir=output_dir,
            bacteria=bacteria,
            background_genes=background_genes,
            fdr_alpha=enrichment_fdr_alpha,
        )
    # =====================================================
    # Step 7. plotting
    # =====================================================
    if make_plots:
        # 网络图
        draw_graph(
            G,
            with_labels=True,
            title=f"{bacteria} Stable Network",
            save_path=os.path.join(output_dir, f"{bacteria}_network.png"),
            show=False
        )
        draw_largest_component(
            G,
            with_labels=True,
            node_size=90,
            font_size=7,
            figsize=(13, 10),
            label_top_n=20,
            title=f"{bacteria} Largest Component",
            save_path=os.path.join(output_dir, f"{bacteria}_largest_component.png"),
            show=False
        )
        draw_graph_by_degree(
            G,
            with_labels=True,
            title=f"{bacteria} Network by Degree",
            save_path=os.path.join(output_dir, f"{bacteria}_network_by_degree.png"),
            show=False
        )
        draw_graph_by_module(
            G,
            membership_df,
            with_labels=True,
            title=f"{bacteria} Network by Module",
            save_path=os.path.join(output_dir, f"{bacteria}_network_by_module.png"),
            label_top_n=20,
            show=False
        )
        draw_largest_component_by_module(
            G,
            membership_df,
            with_labels=True,
            node_size=90,
            font_size=7,
            figsize=(13, 10),
            label_top_n=20,
            top_n_modules=10,
            title=f"{bacteria} Largest Component by Module",
            save_path=os.path.join(output_dir, f"{bacteria}_largest_component_by_module.png"),
            show=False
        )
        _save_module_subgraphs(
            G=G,
            membership_df=membership_df,
            output_dir=output_dir,
            bacteria=bacteria,
            top_n_modules=5
        )
        # 模块大小 / 度分布
        plot_module_size_distribution(
            membership_df,
            save_path=os.path.join(output_dir, f"{bacteria}_module_size_distribution.png"),
            show=False
        )
        plot_degree_distribution(
            G,
            save_path=os.path.join(output_dir, f"{bacteria}_degree_distribution.png"),
            show=False
        )
        plot_degree_distribution_loglog(
            G,
            k_min=3,
            fit_tail=True,
            title=f"{bacteria} Degree Distribution (log-log)",
            save_path=os.path.join(output_dir, f"{bacteria}_degree_distribution_loglog.png"),
            show=False
        )
        # 边权分布
        plot_edge_weight_distribution(
            stable_edges,
            save_path=os.path.join(output_dir, f"{bacteria}_edge_weight_distribution.png"),
            show=False
        )
        plot_edge_weight_density(
            stable_edges,
            save_path=os.path.join(output_dir, f"{bacteria}_edge_weight_density.png"),
            show=False
        )
        # 连通分量大小分布
        plot_component_size_distribution(
            G,
            save_path=os.path.join(output_dir, f"{bacteria}_component_size_distribution.png"),
            show=False
        )
        # top hub 基因图
        plot_top_degree_genes(
            G,
            top_n=20,
            save_path=os.path.join(output_dir, f"{bacteria}_top_degree_genes.png"),
            show=False
        )
        plot_top_weighted_degree_genes(
            G,
            top_n=20,
            save_path=os.path.join(output_dir, f"{bacteria}_top_weighted_degree_genes.png"),
            show=False
        )
        # stability 诊断图
        if not stability_table.empty:
            plot_presence_distribution_combined(
                stability_table,
                n_channels=n_channels,
                save_path=os.path.join(output_dir, f"{bacteria}_presence_count_distribution.png"),
                show=False
            )
            plot_top_stable_edges(
                stability_table,
                top_n=20,
                save_path=os.path.join(output_dir, f"{bacteria}_top_stable_edges.png"),
                show=False
            )

    # =====================================================
    # Step 8. final summary
    # =====================================================
    largest_module_size = 0
    n_modules = 0
    if not module_summary_df.empty:
        n_modules = int(module_summary_df["module_id"].nunique())
        largest_module_size = int(module_summary_df["module_size"].max())

    summary = {
        "bacteria": bacteria,
        "status": "success",
        "n_input_genes": int(expr_df.shape[0]),
        "n_input_samples": int(expr_df.shape[1]),
        "n_preprocessed_genes": int(expr_proc.shape[0]),
        "n_preprocessed_samples": int(expr_proc.shape[1]),
        "n_stable_edges": int(len(stable_edges)),
        "n_nodes": int(graph_summary["n_nodes"]),
        "n_graph_edges": int(graph_summary["n_edges"]),
        "graph_density": float(graph_summary["density"]),
        "average_degree": float(graph_summary["average_degree"]),
        "average_clustering": float(graph_summary["average_clustering"]),
        "n_connected_components": int(graph_summary["n_connected_components"]),
        "largest_component_size": int(graph_summary["largest_component_size"]),
        "n_modules": n_modules,
        "largest_module_size": largest_module_size,
        "method": method,
        "n_channels": int(n_channels),
        "subset_size": runtime_params["subset_size"],
        "n_iterations": runtime_params["n_iterations"],
        "inflation": float(inflation),
        "intersection_mode": intersection_mode,
        "min_presence": int(stability_result["min_presence"]),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"{bacteria}_pipeline_summary.csv"),
        index=False
    )
    if runtime_params["auto_plan"] is not None:
        pd.DataFrame([runtime_params["auto_plan"]]).to_csv(
            os.path.join(output_dir, f"{bacteria}_auto_plan.csv"),
            index=False
        )
    return {
        "expr_proc": expr_proc,
        "stability_result": stability_result,
        "stable_edges": stable_edges,
        "graph": G,
        "clustering_result": clustering_result,
        "module_enrichment_results": module_enrichment_results,
        "summary": summary,
    }

# if __name__ == "__main__":
#     import pandas as pd
#     # =====================================================
#     # 1. 读取真实表达矩阵
#     # 行 = gene, 列 = sample
#     # =====================================================
#     expr_df = pd.read_csv(
#         "tests/data/gene_expression_Acinetobacter_baumannii.csv",
#         index_col=0
#     )
#     expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
#     expr_df = expr_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
#
#     # 如果存在重复基因名，先合并（推荐）
#     expr_df = expr_df.groupby(expr_df.index).mean()
#
#     # =====================================================
#     # 2. 读取 batch 信息
#     # batch 文件至少包含两列：
#     # sample, batch
#     # =====================================================
#     batch_df = pd.read_csv("tests/ab_batch_expanded_clean.csv")
#     batch_series = pd.Series(
#         batch_df["batch"].values,
#         index=batch_df["sample"].values
#     )
#
#     # 对齐到表达矩阵样本顺序
#     batch_series = batch_series.reindex(expr_df.columns)
#
#     if batch_series.isna().any():
#         missing_samples = batch_series[batch_series.isna()].index.tolist()
#         raise ValueError(f"Missing batch labels for samples: {missing_samples}")
#
#     # =====================================================
#     # 3. 运行主流程
#     # =====================================================
#     result = run_stableggm_pipeline(
#         expr_df=expr_df,
#         output_dir="stableggm_ab_output",
#         bacteria="bacteria",
#
#         # preprocess
#         data_type="microarray",
#         normalization=None,   # 如果 preprocess 里会重复做 TPM，建议后续改成 None/skip
#         min_expression=1.0,
#         zero_threshold=0.5,
#         batch_series=batch_series,
#         microarray_logged=False,
#         # stability
#         n_channels=5,
#         subset_size=None,
#         n_iterations=None,
#         random_state=123,
#         iteration_cap=2500,
#         iteration_trigger=3000,
#         max_multiplier=2.0,
#         intersection_mode="soft",
#         min_presence=2,
#
#         # edge selection
#         method="python_genenet_like",
#         fdr_alpha=0.1,
#         prob_threshold=0.9,
#         cutoff_ggm=0.9,
#
#         # clustering
#         inflation=1.2,
#         negative_weight_policy="abs",
#
#         # enrichment
#         # annotation_df=None,
#         enrichment_fdr_alpha=0.05,
#
#         # output / debug
#         save_intermediate=True,
#         make_plots=True,
#         store_pcor_matrices=False,
#         store_edge_lists=False,
#         store_sampled_genes=False,
#     )
#
#     print("Pipeline finished.")
#     print("Summary:")
#     print(result["summary"])
