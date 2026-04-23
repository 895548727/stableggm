import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from sklearn.mixture import GaussianMixture
# =========================================================
# 基础检查
# =========================================================
def _check_matrix(matrix_df: pd.DataFrame):
    if not isinstance(matrix_df, pd.DataFrame):
        raise TypeError("matrix_df must be a pandas DataFrame.")
    if matrix_df.empty:
        raise ValueError("matrix_df is empty.")
    if list(matrix_df.index) != list(matrix_df.columns):
        raise ValueError("matrix_df must have identical row and column labels.")
    if matrix_df.isna().any().any():
        raise ValueError("matrix_df contains NaN values.")
    if not np.allclose(matrix_df.values, matrix_df.values.T, atol=1e-8):
        raise ValueError("matrix_df must be symmetric.")
# =========================================================
# 矩阵转边表（只取上三角）
# =========================================================
def matrix_to_edge_table(
    matrix_df: pd.DataFrame,
    remove_zero_edges: bool = True
) -> pd.DataFrame:
    _check_matrix(matrix_df)
    genes = list(matrix_df.index)
    rows = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            g1 = genes[i]
            g2 = genes[j]
            val = float(matrix_df.iat[i, j])
            if remove_zero_edges and val == 0.0:
                continue
            rows.append({
                "gene1": g1,
                "gene2": g2,
                "weight": val
            })
    return pd.DataFrame(rows)
# =========================================================
# 经验 null + local FDR（GeneNet-like）
# =========================================================
def _fit_empirical_null_gmm(
    z: np.ndarray,
    random_state: int = 42
):
    """
    用双高斯混合模型拟合经验分布：
    - 一个成分作为 null
    - 一个成分作为 signal
    返回：
    - gmm 模型
    - null component index
    - means, stds, pis
    """
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=random_state
    )
    gmm.fit(z)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    pis = gmm.weights_.flatten()
    # 经验上：均值最接近 0 的成分作为 null
    null_idx = np.argmin(np.abs(means))
    return gmm, null_idx, means, stds, pis

def _compute_lfdr_from_gmm(
    z: np.ndarray,
    gmm,
    null_idx: int,
    means: np.ndarray,
    stds: np.ndarray,
    pis: np.ndarray
):
    """
    根据双高斯混合模型计算 local FDR:
        lfdr(x) = pi0 * f0(x) / f(x)
    """
    z = np.asarray(z, dtype=float)
    # 总密度 f(x)
    total_density = np.zeros_like(z, dtype=float)
    for k in range(len(means)):
        total_density += pis[k] * norm.pdf(z, loc=means[k], scale=stds[k])
    # null 密度 pi0 * f0(x)
    null_density = pis[null_idx] * norm.pdf(
        z,
        loc=means[null_idx],
        scale=stds[null_idx]
    )
    lfdr = null_density / np.maximum(total_density, 1e-300)
    lfdr = np.clip(lfdr, 0.0, 1.0)
    return lfdr

# =========================================================
# 类似 GeneNet 的边筛选
# =========================================================
def cal_net_python_genenet_like(
    matrix_df: pd.DataFrame,
    prob_threshold: float = 0.9,
    fdr_alpha: float = 0.1,
    random_state: int = 42,
    weight_threshold: float = 0.1
) -> pd.DataFrame:
    """
    类似 GeneNet/fdrtool 的 Python 版边筛选（近似实现）：
    1. 将矩阵转为 edge table
    2. 用经验 null（双高斯混合）估计 local FDR
    3. 输出 pvalue / qvalue / prob = 1-lfdr
    4. 按 prob_threshold 和 fdr_alpha 筛边
    注意：
    ----
    matrix_df 建议输入已经做过 Fisher z + mean-centering 的对称矩阵。
    这不是 GeneNet 的严格复现，而是 GeneNet-like 近似方案。
    """
    _check_matrix(matrix_df)
    edge_df = matrix_to_edge_table(matrix_df, remove_zero_edges=True)
    if edge_df.empty:
        return pd.DataFrame(
            columns=[
                "gene1", "gene2", "weight", "zscore",
                "pvalue", "qvalue", "lfdr", "prob"
            ]
        )
    zscore = edge_df["weight"].values.astype(float)
    # Step 1: 经验 null 拟合
    gmm, null_idx, means, stds, pis = _fit_empirical_null_gmm(
        zscore,
        random_state=random_state
    )
    null_mean = means[null_idx]
    null_sd = stds[null_idx]
    # Step 2: 用经验 null 计算双侧 p 值
    z_standardized = (zscore - null_mean) / np.maximum(null_sd, 1e-12)
    pvals = 2 * (1 - norm.cdf(np.abs(z_standardized)))
    # Step 3: BH q-value
    _, qvals, _, _ = multipletests(pvals, alpha=fdr_alpha, method="fdr_bh")
    # Step 4: local FDR
    lfdr = _compute_lfdr_from_gmm(
        zscore,
        gmm=gmm,
        null_idx=null_idx,
        means=means,
        stds=stds,
        pis=pis
    )
    prob = 1.0 - lfdr
    edge_df["zscore"] = zscore
    edge_df["pvalue"] = pvals
    edge_df["qvalue"] = qvals
    edge_df["lfdr"] = lfdr
    edge_df["prob"] = prob
    # 类似 GeneNet 的筛选：prob 高、qvalue 小
    net_df = edge_df.loc[
        (edge_df["prob"] >= prob_threshold) &
        (edge_df["qvalue"] <= fdr_alpha) &
        (edge_df["weight"].abs() >= weight_threshold)
        ].copy()
    net_df = net_df.sort_values(
        by=["prob", "gene1", "gene2"],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    return net_df
# =========================================================
# 统一接口
# =========================================================
def select_edges(
    matrix_df: pd.DataFrame,
    method: str = "python_genenet_like",
    **kwargs
) -> pd.DataFrame:
    if method == "python_genenet_like":
        return cal_net_python_genenet_like(matrix_df, **kwargs)
    else:
        raise ValueError("method must be 'python_genenet_like'")