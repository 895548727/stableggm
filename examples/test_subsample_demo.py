import pandas as pd
import numpy as np
import pycombat
import time
from stableggm.subsample import run_subsample_pcor, concat_edge_lists


# =========================================================
# 1. 读入 TPM 数据
# =========================================================
expr_df = pd.read_csv("../tests/data/gene_expression_Acinetobacter_baumannii.csv", index_col=0)
expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
expr_df = expr_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

# =========================================================
# 2. 读入 batch 信息
# =========================================================
batch_df = pd.read_csv("../tests/ab_batch_expanded_clean.csv")
batch_series = pd.Series(batch_df["batch"].values, index=batch_df["sample"].values)
batch_series = batch_series.reindex(expr_df.columns)

if batch_series.isna().any():
    missing_samples = batch_series[batch_series.isna()].index.tolist()
    raise ValueError(f"Missing batch labels for samples: {missing_samples}")

# =========================================================
# 3. 与 preprocess.py 保持一致的前处理
#    这里用“适合 TPM 的版本”：过滤 + log1p + ComBat
# =========================================================
zero_threshold = 0.5
mask = (expr_df == 0).sum(axis=1) <= (zero_threshold * expr_df.shape[1])
expr_filtered = expr_df.loc[mask].copy()

expr_preprocessed = np.log1p(expr_filtered)

combat = pycombat.Combat()
expr_corrected = combat.fit_transform(expr_preprocessed.T.values, batch_series.values)
expr_preprocessed = pd.DataFrame(
    expr_corrected,
    index=expr_preprocessed.columns,
    columns=expr_preprocessed.index
).T

print("Preprocessed expression shape:", expr_preprocessed.shape)

# =========================================================
# 4. 为了调试，先只取前 100 个基因
#    避免太慢，也方便看结果
# =========================================================
expr_preprocessed = expr_preprocessed.groupby(expr_preprocessed.index).mean()
expr_input = expr_preprocessed.copy()
print("Input expression shape:", expr_input.shape)
start_time = time.time()
results = run_subsample_pcor(
    expr_df=expr_input,
    subset_size=None,
    n_iterations=None,
    random_state=42,
    store_pcor_matrices=True,
    store_edge_lists=True,
    store_sampled_genes=True,
)
# =========================================================
# 记录结束时间
# =========================================================
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60

# =========================================================
# 取最终结果
# =========================================================
aggregated_edge_df = results["aggregated_edge_df"].copy()

# 可选：加一列 presence_ratio
aggregated_edge_df["presence_ratio"] = (
    aggregated_edge_df["n_occurrence"] / results["n_iterations"]
)

# =========================================================
# 输出结果
# =========================================================
aggregated_edge_df.to_csv("./aggregated_edge_df.csv", index=False)

# 同时把运行摘要也保存下来
summary_df = pd.DataFrame([{
    "n_genes_input": expr_input.shape[0],
    "n_samples_input": expr_input.shape[1],
    "subset_size": results["subset_size"],
    "n_iterations": results["n_iterations"],
    "elapsed_seconds": elapsed_seconds,
    "elapsed_minutes": elapsed_minutes,
    "n_edges_final": aggregated_edge_df.shape[0],
    "strategy": results["auto_plan"]["strategy"] if results["auto_plan"] is not None else "manual"
}])

# =========================================================
# 打印摘要
# =========================================================
print("\n===== Run summary =====")
print("Input expression shape:", expr_input.shape)
print("subset_size:", results["subset_size"])
print("n_iterations:", results["n_iterations"])
print("final aggregated_edge_df shape:", aggregated_edge_df.shape)
print(f"Elapsed time: {elapsed_seconds:.2f} seconds ({elapsed_minutes:.2f} minutes)")
print("Saved:")
print(" - ./aggregated_edge_df.csv")
print(" - ./subsample_run_summary.csv")