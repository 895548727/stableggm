import pandas as pd
from stableggm.preprocess import preprocess_expression
from stableggm.pcor import compute_partial_correlation, pcor_to_edge_list

# 原始表达矩阵
expr_df = pd.DataFrame({
    "sample1": [10, 0, 5, 0],
    "sample2": [15, 0, 3, 0],
    "sample3": [12, 0, 4, 0],
    "sample4": [11, 0, 6, 0]
}, index=["gene1", "gene2", "gene3", "gene4"])

print("原始矩阵：")
print(expr_df)

# 预处理
expr_proc = preprocess_expression(
    expr_df,
    data_type="RNA-seq",
    normalization="CPM"
)

print("\n预处理后矩阵：")
print(expr_proc)

# 计算 partial correlation
pcor_df = compute_partial_correlation(expr_proc)

print("\nPartial correlation matrix：")
print(pcor_df)

# 转边表
edges_df = pcor_to_edge_list(pcor_df, abs_threshold=0.0)

print("\nEdge list：")
print(edges_df)