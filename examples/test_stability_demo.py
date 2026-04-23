import pandas  as pd
from stableggm.pipeline import run_stableggm_pipeline

expr_df = pd.read_csv(
        "../tests/data/gene_expression_Acinetobacter_baumannii_2.csv",
        index_col=0
    )

expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
expr_df = expr_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

 # 如果存在重复基因名，先合并（推荐）
expr_df = expr_df.groupby(expr_df.index).mean()

# =====================================================
# 2. 读取 batch 信息
# batch 文件至少包含两列：
# sample, batch
# =====================================================
batch_df = pd.read_csv("../tests/ab_batch_expanded_clean.csv")
batch_series = pd.Series(
        batch_df["batch"].values,
        index=batch_df["sample"].values
    )
# 对齐到表达矩阵样本顺序
batch_series = batch_series.reindex(expr_df.columns)

if batch_series.isna().any():
    missing_samples = batch_series[batch_series.isna()].index.tolist()
    raise ValueError(f"Missing batch labels for samples: {missing_samples}")

anno_df = pd.read_csv("../tests/ab_output_go_kegg_mapped.tsv", sep="\t")
# =====================================================
# 3. 运行主流程
# =====================================================
result = run_stableggm_pipeline(
    expr_df=expr_df,
    output_dir="stableggm_ab_output",
    bacteria="Acinetobacter_baumannii",

    # preprocess
    data_type="microarray",
    normalization=None,   # 如果 preprocess 里会重复做 TPM，建议后续改成 None/skip
    # min_expression=1.0,
    zero_threshold=0.5,
    batch_series=batch_series,
    microarray_logged=False,

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
    # 新增：画多通道 venn 图
    plot_venn=True,
    plot_inflation_sensitivity=True,

    # edge selection
    method="python_genenet_like",
    fdr_alpha=0.1,
    prob_threshold=0.9,
    cutoff_ggm=0.9,

    # clustering
    inflation=1.2,
    run_inflation_scan=True,
    inflation_values=[1.4, 1.6, 1.8, 2.0, 2.2],
    # enrichment
    annotation_df=anno_df,
    enrichment_fdr_alpha=0.05,

    # output / debug
    save_intermediate=True,
    make_plots=True,
    store_pcor_matrices=False,
    store_edge_lists=False,
    store_sampled_genes=False,
)
print("Pipeline finished.")
print("Summary:")
print(result["summary"])