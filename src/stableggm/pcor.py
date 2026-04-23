import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def _validate_expression_matrix(expr_df: pd.DataFrame) -> pd.DataFrame:
    """
    检查输入表达矩阵是否合法。
    默认要求:
    - 行: gene
    - 列: sample
    - 数值型
    - 不含 NA
    """
    if not isinstance(expr_df, pd.DataFrame):
        raise TypeError("expr_df must be a pandas DataFrame.")
    if expr_df.empty:
        raise ValueError("expr_df is empty.")
    if expr_df.shape[0] < 2:
        raise ValueError("expr_df must contain at least 2 genes.")
    if expr_df.shape[1] < 2:
        raise ValueError("expr_df must contain at least 2 samples.")
    # 强制转为 float
    expr_df = expr_df.astype(float)
    if expr_df.isna().any().any():
        raise ValueError("expr_df contains NaN values. Please preprocess first.")
    return expr_df

def compute_precision_matrix(expr_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 precision matrix（逆协方差矩阵）。
    参数
    ----
    expr_df : pd.DataFrame
        行 = gene, 列 = sample
    返回
    ----
    precision_df : pd.DataFrame
        gene × gene 的 precision matrix
    """
    expr_df = _validate_expression_matrix(expr_df)
    # sklearn 默认: 行=样本, 列=特征
    # 当前 expr_df 是 行=gene, 列=sample
    # 所以转置后: 行=sample, 列=gene
    X = expr_df.T.values
    lw = LedoitWolf()
    lw.fit(X)
    precision = lw.precision_
    precision_df = pd.DataFrame(
        precision,
        index=expr_df.index,
        columns=expr_df.index
    )
    return precision_df

def compute_partial_correlation(expr_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 partial correlation matrix。
    参数
    ----
    expr_df : pd.DataFrame
        行 = gene, 列 = sample
    返回
    ----
    pcor_df : pd.DataFrame
        gene × gene 的 partial correlation matrix
    """
    precision_df = compute_precision_matrix(expr_df)
    precision = precision_df.values
    diag = np.diag(precision)

    if np.any(diag <= 0):
        raise ValueError(
            "Precision matrix has non-positive diagonal entries; "
            "cannot compute partial correlation safely."
        )
    d = np.sqrt(diag)
    denom = np.outer(d, d)
    pcor = -precision / denom
    # 对角线设为 1
    np.fill_diagonal(pcor, 1.0)
    pcor_df = pd.DataFrame(
        pcor,
        index=precision_df.index,
        columns=precision_df.columns
    )
    return pcor_df

def pcor_to_edge_list(pcor_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 partial correlation matrix 转成 edge list。

    参数
    ----
    pcor_df : pd.DataFrame
        gene × gene 的 partial correlation matrix

    返回
    ----
    edges_df : pd.DataFrame
        三列: gene1, gene2, pcor
    """
    if not isinstance(pcor_df, pd.DataFrame):
        raise TypeError("pcor_df must be a pandas DataFrame.")

    if pcor_df.empty:
        return pd.DataFrame(columns=["gene1", "gene2", "pcor"])

    if list(pcor_df.index) != list(pcor_df.columns):
        raise ValueError("pcor_df must have identical row and column labels.")

    if not np.allclose(pcor_df.values, pcor_df.values.T, atol=1e-8):
        raise ValueError("pcor_df must be symmetric.")

    genes = list(pcor_df.index)
    edges = []

    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            value = float(pcor_df.iat[i, j])
            edges.append((genes[i], genes[j], value))

    edges_df = pd.DataFrame(edges, columns=["gene1", "gene2", "pcor"])
    return edges_df