import time
import tracemalloc
from pathlib import Path
import pandas as pd
from stableggm.pipeline import run_stableggm_pipeline

def load_expression_and_batch():
    expr_df = pd.read_csv(
        "../tests/data/gene_expression_Acinetobacter_baumannii_2.csv",
        index_col=0
    )
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
    expr_df = expr_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    expr_df = expr_df.groupby(expr_df.index).mean()
    batch_df = pd.read_csv("../tests/ab_batch_expanded_clean.csv")
    batch_series = pd.Series(
        batch_df["batch"].values,
        index=batch_df["sample"].values
    )
    batch_series = batch_series.reindex(expr_df.columns)
    if batch_series.isna().any():
        missing_samples = batch_series[batch_series.isna()].index.tolist()
        raise ValueError(f"Missing batch labels for samples: {missing_samples}")
    return expr_df, batch_series

def run_one_benchmark(expr_sub: pd.DataFrame, batch_series: pd.Series) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()
    result = run_stableggm_pipeline(
        expr_df=expr_sub,
        output_dir="benchmark_output" ,                   # 不写文件，避免 I/O 干扰
        bacteria="Acinetobacter_baumannii",
        # preprocess
        data_type="microarray",
        normalization=None,
        zero_threshold=0.5,
        batch_series=batch_series,
        microarray_logged=False,
        expr_before_batch=None,
        expr_after_batch=None,
        # stability
        n_channels=3,
        subset_size=None,
        n_iterations=None,
        random_state=123,
        iteration_cap=2500,
        iteration_trigger=3000,
        max_multiplier=2.0,
        intersection_mode="soft",
        min_presence=2,
        plot_venn=False,
        plot_inflation_sensitivity=False,
        # edge selection
        method="python_genenet_like",
        fdr_alpha=0.1,
        prob_threshold=0.9,
        cutoff_ggm=0.9,
        # clustering
        inflation=1.2,
        run_inflation_scan=False,
        inflation_values=None,
        # enrichment
        annotation_df=None,
        enrichment_fdr_alpha=0.05,
        # output / debug
        save_intermediate=False,
        make_plots=False,
        store_pcor_matrices=False,
        store_edge_lists=False,
        store_sampled_genes=False,
    )
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "runtime_sec": elapsed,
        "peak_mem_mb": peak / 1024 / 1024,
        "summary": result.get("summary", {})
    }

def benchmark_by_gene_count():
    expr_df, batch_series = load_expression_and_batch()
    # 固定样本数，只改变基因数
    # 建议按你数据规模自行调整
    gene_points = [700, 1400, 2100, 2800, 3150]
    rows = []
    for n_genes in gene_points:
        if n_genes > expr_df.shape[0]:
            print(f"[WARN] Skip {n_genes}: only {expr_df.shape[0]} genes available.")
            continue

        expr_sub = expr_df.iloc[:n_genes, :].copy()

        print(f"[INFO] Running benchmark for {n_genes} genes × {expr_sub.shape[1]} samples ...")
        out = run_one_benchmark(expr_sub, batch_series)

        row = {
            "n_genes": n_genes,
            "n_samples": expr_sub.shape[1],
            "runtime_sec": out["runtime_sec"],
            "peak_mem_mb": out["peak_mem_mb"],
        }

        summary = out["summary"]
        if isinstance(summary, dict):
            row["n_stable_edges"] = summary.get("n_stable_edges")
            row["n_nodes"] = summary.get("n_nodes")
            row["n_modules"] = summary.get("n_modules")

        rows.append(row)

    result_df = pd.DataFrame(rows)
    print("\nBenchmark results:")
    print(result_df)

    result_df.to_csv("stableggm_runtime_memory_benchmark.csv", index=False)
    print("\nSaved to stableggm_runtime_memory_benchmark.csv")


if __name__ == "__main__":
    benchmark_by_gene_count()