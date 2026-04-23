import os
import time
import pandas as pd

from stableggm.pipeline import run_stableggm_pipeline
from stableggm.plotting import (
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
    plot_presence_count_distribution,
    plot_presence_ratio_distribution,
    plot_top_stable_edges,
)


def load_expression_matrix(expr_path: str) -> pd.DataFrame:
    expr_df = pd.read_csv(expr_path, index_col=0)

    if expr_df.empty:
        raise ValueError("Expression matrix is empty.")

    print(f"[INFO] Loaded expression matrix: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
    return expr_df


def load_batch_series(batch_path: str | None) -> pd.Series | None:
    if batch_path is None:
        return None

    batch_df = pd.read_csv(batch_path)
    required_cols = {"sample", "batch"}
    missing = required_cols - set(batch_df.columns)
    if missing:
        raise ValueError(f"Batch file is missing required columns: {missing}")

    batch_series = pd.Series(batch_df["batch"].values, index=batch_df["sample"].astype(str).values)
    print(f"[INFO] Loaded batch labels for {len(batch_series)} samples")
    return batch_series


def load_annotation_df(annotation_path: str | None) -> pd.DataFrame | None:
    if annotation_path is None:
        return None

    annotation_df = pd.read_csv(annotation_path)
    required_cols = {"gene", "term"}
    missing = required_cols - set(annotation_df.columns)
    if missing:
        raise ValueError(f"Annotation file is missing required columns: {missing}")

    print(f"[INFO] Loaded annotation table with {len(annotation_df)} gene-term rows")
    return annotation_df


def make_smoke_subset(
    expr_df: pd.DataFrame,
    n_genes: int = 500,
    n_samples: int = 50
) -> pd.DataFrame:
    n_genes = min(n_genes, expr_df.shape[0])
    n_samples = min(n_samples, expr_df.shape[1])

    expr_small = expr_df.iloc[:n_genes, :n_samples].copy()
    print(f"[INFO] Smoke subset: {expr_small.shape[0]} genes × {expr_small.shape[1]} samples")
    return expr_small


def align_batch_series_to_expr(batch_series: pd.Series | None, expr_df: pd.DataFrame) -> pd.Series | None:
    if batch_series is None:
        return None

    sample_names = list(map(str, expr_df.columns))
    missing = [s for s in sample_names if s not in batch_series.index]
    if missing:
        raise ValueError(
            f"Batch series is missing {len(missing)} samples from expression matrix, e.g. {missing[:5]}"
        )

    return batch_series.loc[sample_names]


def save_extra_plots(result: dict, plots_dir: str, bacteria: str):
    """
    利用 pipeline 返回结果，再额外补画一组图，统一保存到 results/.../plots
    """
    os.makedirs(plots_dir, exist_ok=True)

    G = result.get("graph", None)
    stability_result = result.get("stability_result", None)
    stable_edges = result.get("stable_edges", None)
    clustering_result = result.get("clustering_result", None)

    if stability_result is not None:
        stability_table = stability_result.get("stability_table", pd.DataFrame())
        if not stability_table.empty:
            plot_presence_count_distribution(
                stability_table,
                save_path=os.path.join(plots_dir, f"{bacteria}_presence_count_distribution_extra.png"),
                show=False,
            )
            plot_presence_ratio_distribution(
                stability_table,
                save_path=os.path.join(plots_dir, f"{bacteria}_presence_ratio_distribution_extra.png"),
                show=False,
            )
            plot_top_stable_edges(
                stability_table,
                top_n=20,
                save_path=os.path.join(plots_dir, f"{bacteria}_top_stable_edges_extra.png"),
                show=False,
            )

    if G is not None and G.number_of_nodes() > 0:
        draw_graph(
            G,
            with_labels=False,
            title=f"{bacteria} Stable Network (extra)",
            save_path=os.path.join(plots_dir, f"{bacteria}_network_extra.png"),
            show=False,
        )

        draw_largest_component(
            G,
            with_labels=False,
            title=f"{bacteria} Largest Component (extra)",
            save_path=os.path.join(plots_dir, f"{bacteria}_largest_component_extra.png"),
            show=False,
        )

        draw_graph_by_degree(
            G,
            with_labels=False,
            title=f"{bacteria} Network by Degree (extra)",
            save_path=os.path.join(plots_dir, f"{bacteria}_network_by_degree_extra.png"),
            show=False,
        )

        plot_degree_distribution(
            G,
            save_path=os.path.join(plots_dir, f"{bacteria}_degree_distribution_extra.png"),
            show=False,
        )

        plot_component_size_distribution(
            G,
            save_path=os.path.join(plots_dir, f"{bacteria}_component_size_distribution_extra.png"),
            show=False,
        )

        plot_top_degree_genes(
            G,
            top_n=20,
            save_path=os.path.join(plots_dir, f"{bacteria}_top_degree_genes_extra.png"),
            show=False,
        )

        plot_top_weighted_degree_genes(
            G,
            top_n=20,
            save_path=os.path.join(plots_dir, f"{bacteria}_top_weighted_degree_genes_extra.png"),
            show=False,
        )

    if stable_edges is not None and not stable_edges.empty:
        if "weight" in stable_edges.columns:
            plot_edge_weight_distribution(
                stable_edges,
                weight_col="weight",
                save_path=os.path.join(plots_dir, f"{bacteria}_edge_weight_distribution_extra.png"),
                show=False,
            )
            plot_edge_weight_density(
                stable_edges,
                weight_col="weight",
                save_path=os.path.join(plots_dir, f"{bacteria}_edge_weight_density_extra.png"),
                show=False,
            )

    if G is not None and clustering_result is not None:
        membership_df = clustering_result.get("membership_df", pd.DataFrame())

        if not membership_df.empty:
            draw_graph_by_module(
                G,
                membership_df,
                with_labels=False,
                title=f"{bacteria} Network by Module (extra)",
                save_path=os.path.join(plots_dir, f"{bacteria}_network_by_module_extra.png"),
                show=False,
            )

            draw_largest_component_by_module(
                G,
                membership_df,
                with_labels=False,
                title=f"{bacteria} Largest Component by Module (extra)",
                save_path=os.path.join(plots_dir, f"{bacteria}_largest_component_by_module_extra.png"),
                show=False,
            )

            plot_module_size_distribution(
                membership_df,
                save_path=os.path.join(plots_dir, f"{bacteria}_module_size_distribution_extra.png"),
                show=False,
            )

            # 额外保存前 3 个最大模块的子图
            module_sizes = (
                membership_df.groupby("module_id")["gene"]
                .count()
                .sort_values(ascending=False)
            )
            top_modules = list(module_sizes.head(3).index)

            for module_id in top_modules:
                draw_module_subgraph(
                    G,
                    membership_df,
                    module_id=module_id,
                    with_labels=True,
                    title=f"{bacteria} Module {module_id} (extra)",
                    save_path=os.path.join(plots_dir, f"{bacteria}_module_{module_id}_subgraph_extra.png"),
                    show=False,
                )


def main():
    # =====================================================
    # 1. 路径设置
    # =====================================================
    expr_path = "data/gene_expression_Acinetobacter_baumannii.csv"
    batch_path = "ab_batch_expanded_clean.csv"
    annotation_path = None

    # =====================================================
    # 2. 模式
    # =====================================================
    mode = "full"
    # mode = "smoke"

    smoke_n_genes = 500
    smoke_n_samples = 50

    bacteria = "Acinetobacter_baumannii"

    # =====================================================
    # 3. 输出目录改到 results 下
    # =====================================================
    base_output_dir = os.path.join("results", bacteria)
    csv_dir = os.path.join(base_output_dir, "csv")
    plots_dir = os.path.join(base_output_dir, "plots")

    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # =====================================================
    # 4. 计时开始
    # =====================================================
    start_time = time.time()
    print("=" * 60)
    print("StableGGM Pipeline - Full Run")
    print(f"Mode: {mode}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # =====================================================
    # 5. 读取数据
    # =====================================================
    expr_df = load_expression_matrix(expr_path)
    batch_series = load_batch_series(batch_path)
    annotation_df = load_annotation_df(annotation_path)

    # =====================================================
    # 6. 选择测试数据
    # =====================================================
    if mode == "smoke":
        expr_input = make_smoke_subset(
            expr_df,
            n_genes=smoke_n_genes,
            n_samples=smoke_n_samples
        )
    elif mode == "full":
        expr_input = expr_df.copy()
        print(f"[INFO] Full dataset mode: {expr_input.shape[0]} genes × {expr_input.shape[1]} samples")
    else:
        raise ValueError("mode must be 'smoke' or 'full'")

    batch_input = align_batch_series_to_expr(batch_series, expr_input)

    if annotation_df is not None:
        genes_in_expr = set(map(str, expr_input.index))
        annotation_df = annotation_df.copy()
        annotation_df["gene"] = annotation_df["gene"].astype(str)
        annotation_df = annotation_df.loc[
            annotation_df["gene"].isin(genes_in_expr)
        ].reset_index(drop=True)
        print(f"[INFO] Filtered annotation table to {len(annotation_df)} rows matching current expression genes")

    # =====================================================
    # 7. 跑 pipeline
    # 注意：这里 output_dir 指向 csv_dir，pipeline 本身也会保存图和表
    # =====================================================
    result = run_stableggm_pipeline(
        expr_df=expr_input,
        output_dir=csv_dir,
        bacteria=bacteria,

        # preprocess
        data_type="RNA-seq",
        normalization="CPM",
        min_expression=1.0,
        zero_threshold=0.5,
        batch_series=batch_input,
        microarray_logged=True,

        expr_before_batch=None,
        expr_after_batch=None,

        # stability / subsample
        n_channels=5 if mode == "full" else 3,
        subset_size=None,
        n_iterations=None,
        random_state=123,
        iteration_cap=2500 if mode == "full" else 500,
        iteration_trigger=3000 if mode == "full" else 800,
        max_multiplier=2.0,
        intersection_mode="soft",
        min_presence=2,

        # edge selection
        method="python_bh",
        fdr_alpha=0.1,
        score_threshold=0.9,
        cutoff_ggm=0.9,

        # clustering
        inflation=2.0,
        negative_weight_policy="abs",

        # enrichment
        annotation_df=annotation_df,
        enrichment_fdr_alpha=0.05,

        # output / debug
        save_intermediate=True,
        make_plots=True,
        store_pcor_matrices=False,
        store_edge_lists=False,
        store_sampled_genes=False,
    )

    # =====================================================
    # 8. 额外画图并统一保存到 results/.../plots
    # =====================================================
    save_extra_plots(
        result=result,
        plots_dir=plots_dir,
        bacteria=bacteria
    )

    # =====================================================
    # 9. 保存 summary 到 results 根目录
    # =====================================================
    summary_df = pd.DataFrame([result["summary"]])
    summary_df.to_csv(os.path.join(base_output_dir, f"{bacteria}_summary.csv"), index=False)

    # =====================================================
    # 10. 打印 summary 和运行时间
    # =====================================================
    print("\n[INFO] Pipeline finished.")
    print("[INFO] Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")

    end_time = time.time()
    elapsed = end_time - start_time
    print("\n" + "=" * 60)
    print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Results saved under: {base_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()