import numpy as np
import pandas as pd

def fisher_z_transform(values, eps: float = 1e-12):
    """
    Fisher z-transformation:
        z = 0.5 * ln((1+r)/(1-r)) = arctanh(r)
    参数
    ----
    values : array-like
        偏相关系数，理论上应在 (-1, 1) 内
    eps : float
        防止 r = ±1 导致无穷大

    返回
    ----
    np.ndarray
        Fisher z 变换后的数值
    """
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -1 + eps, 1 - eps)
    return np.arctanh(values)

def normalize_pcor_column(
    edge_df: pd.DataFrame,
    pcor_col: str = "pcor"
) -> pd.DataFrame:
    """
    对聚合后的 edge table 中的 pcor 列做：
    1. Fisher z-transform
    2. mean-centering
    参数
    ----
    edge_df : pd.DataFrame
        至少包含列:
        - gene1
        - gene2
        - pcor_col
    pcor_col : str
        需要标准化的偏相关列名，默认 'pcor'
    返回
    ----
    pd.DataFrame
        在原表基础上新增两列：
        - z_pcor
        - norm_pcor
    """
    if not isinstance(edge_df, pd.DataFrame):
        raise TypeError("edge_df must be a pandas DataFrame.")
    required_cols = {"gene1", "gene2", pcor_col}
    missing = required_cols - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge_df is missing required columns: {missing}")
    if edge_df.empty:
        return edge_df.copy()
    df = edge_df.copy()
    df["gene1"] = df["gene1"].astype(str)
    df["gene2"] = df["gene2"].astype(str)
    df[pcor_col] = df[pcor_col].astype(float)
    # Fisher z-transform
    z = fisher_z_transform(df[pcor_col].values)
    # mean-centering
    z_mean = z.mean()
    norm = z - z_mean
    df["z_pcor"] = z
    df["norm_pcor"] = norm
    return df