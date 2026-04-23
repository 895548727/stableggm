import numpy as np
import pandas as pd

from stableggm.preprocess import preprocess_expression
from stableggm.subsample import run_subsample_pcor, concat_edge_lists
# from stableggm.aggregate import aggregate_edges_min_abs, filter_edges_by_pcor_threshold

# 构造测试矩阵
np.random.seed(42)

genes = [f"gene{i}" for i in range(1, 11)]
samples = [f"sample{j}" for j in range(1, 6)]

expr_df = pd.DataFrame(
    np.random.poisson(lam=10, size=(10, 5)),
    index=genes,
    columns=samples
)

print("原始矩阵：")
print(expr_df)

expr_proc = preprocess_expression(
    expr_df,
    data_type="RNA-seq",
    normalization="CPM"
)

print("\n预处理后矩阵：")
print(expr_proc)

results = run_subsample_pcor(
    expr_df=expr_proc,
    subset_size=5,
    n_iterations=5,
    abs_threshold=0.0,
    random_state=123
)

all_edges = concat_edge_lists(results["edge_lists"])

print("\n拼接后的所有边：")
print(all_edges.head(20))

aggregated_edges = aggregate_edges_min_abs(all_edges)

print("\n聚合后的边：")
print(aggregated_edges.head(20))

final_edges = filter_edges_by_pcor_threshold(
    aggregated_edges,
    threshold=0.1
)

print("\n最终保留的边（|pcor| >= 0.1）：")
print(final_edges.head(20))