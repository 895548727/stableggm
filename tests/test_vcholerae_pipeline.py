import os
import pandas as pd
from stableggm.pipeline import run_stableggm_pipeline

def load_expression_matrix(expr_path: str) -> pd.DataFrame:
    """
    读取表达矩阵。
    要求：
    - 第一列为 gene 名
    - 后续列为 sample
    - 行 = gene，列 = sample
    """
    expr_df = pd.read_csv(expr_path, index_col=0)

    if expr_df.empty:
        raise ValueError("Expression matrix is empty.")

    print(f"[INFO] Loaded expression matrix: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
    return expr_df

def load_batch_series(batch_path: str | None) -> pd.Series | None:
    """
    读取批次信息。
    CSV 格式要求至少两列：
    - sample
    - batch
    """
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
    """
    读取富集分析注释表。
    CSV 格式要求至少两列：
    - gene
    - term
    """
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
    """
    从真实表达矩阵中截取一个小规模子集用于 smoke test。
    默认取前 500 个基因、前 50 个样本。
    """
    n_genes = min(n_genes, expr_df.shape[0])
    n_samples = min(n_samples, expr_df.shape[1])

    expr_small = expr_df.iloc[:n_genes, :n_samples].copy()
    print(f"[INFO] Smoke subset: {expr_small.shape[0]} genes × {expr_small.shape[1]} samples")
    return expr_small

def align_batch_series_to_expr(batch_series: pd.Series | None, expr_df: pd.DataFrame) -> pd.Series | None:
    """
    将 batch_series 对齐到表达矩阵列名。
    """
    if batch_series is None:
        return None

    sample_names = list(map(str, expr_df.columns))
    missing = [s for s in sample_names if s not in batch_series.index]
    if missing:
        raise ValueError(
            f"Batch series is missing {len(missing)} samples from expression matrix, e.g. {missing[:5]}"
        )

    return batch_series.loc[sample_names]

def main():
    # =====================================================
    # 1. 用户需要修改的路径
    # =====================================================
    expr_path = "data/gene_expression_Acinetobacter_baumannii.csv"
    batch_path = "ab_batch_expanded_clean.csv"
    annotation_path = None

    # =====================================================
    # 2. 测试模式
    # mode = "smoke" 先小规模跑通
    # mode = "full"  跑全量数据
    # =====================================================
    mode = "smoke"

    # smoke test 子集大小
    smoke_n_genes = 500
    smoke_n_samples = 50

    # 输出目录
    output_dir = "vcholerae_test_output"

    # =====================================================
    # 3. 读取数据
    # =====================================================
    expr_df = load_expression_matrix(expr_path)
    batch_series = load_batch_series(batch_path)
    annotation_df = load_annotation_df(annotation_path)

    # =====================================================
    # 4. 选择测试数据
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
        annotation_df = annotation_df.loc[annotation_df["gene"].isin(genes_in_expr)].reset_index(drop=True)
        print(f"[INFO] Filtered annotation table to {len(annotation_df)} rows matching current expression genes")

    # =====================================================
    # 5. 创建输出目录
    # =====================================================
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # 6. 跑 pipeline
    # =====================================================
    result = run_stableggm_pipeline(
        expr_df=expr_input,
        output_dir=output_dir,
        bacteria="vcholerae",

        # preprocess
        data_type="RNA-seq",
        normalization="CPM",
        min_expression=1.0,
        zero_threshold=0.5,
        batch_series=batch_input,
        microarray_logged=True,

        # 如果你已经另外保存了去批次前后矩阵，可以传进来画箱线图
        expr_before_batch=None,
        expr_after_batch=None,

        # stability / subsample
        n_channels=3 if mode == "smoke" else 5,
        subset_size=None,
        n_iterations=None,
        random_state=123,
        iteration_cap=500 if mode == "smoke" else 2500,
        iteration_trigger=800 if mode == "smoke" else 3000,
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
    # 7. 打印 summary
    # =====================================================
    print("\n[INFO] Pipeline finished.")
    print("[INFO] Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()