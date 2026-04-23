from __future__ import annotations
import numpy as np
import pandas as pd

def preprocess_expression(
    expr_df: pd.DataFrame,
    data_type="RNA-seq",
    normalization="DESeq2",
    min_expression=None,
    batch_series=None,
    zero_threshold=0.5,
    microarray_logged=False,
):
    """
    Support basic filtering, normalization, optional batch correction, and log transforms.

    Returns
    -------
    dict with:
        - expr_preprocessed
        - expr_before_batch
        - expr_after_batch
    """
    df = expr_df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if min_expression is not None:
        df = df.loc[df.mean(axis=1) >= float(min_expression)]

    mask = (df == 0).sum(axis=1) <= (zero_threshold * df.shape[1])
    df = df[mask]

    if data_type == "RNA-seq":
        if normalization == "DESeq2":
            geometric_means = np.exp(np.log(df + 1e-6).mean(axis=1))
            size_factors = df.divide(geometric_means, axis=0).median(axis=0)
            df = df.divide(size_factors, axis=1)
        elif normalization == "CPM":
            df = df.divide(df.sum(axis=0), axis=1) * 1e6
        elif normalization == "FPM":
            df = df.divide(df.sum(axis=0), axis=1) * 1e6
    elif data_type == "microarray":
        pass
    # 先记录 batch correction 前矩阵（未 log）
    expr_before_batch = df.copy()
    if batch_series is not None:
        try:
            import pycombat
        except ImportError as exc:
            raise ImportError(
                "pycombat is required when batch_series is provided. "
                "Install it with `pip install pycombat`."
            ) from exc
        batch_series = pd.Series(batch_series, index=df.columns)
        if len(batch_series) != df.shape[1]:
            raise ValueError("batch_series length does not match number of samples.")
        expr = df.T
        batch_aligned = batch_series.reindex(expr.index).values
        if pd.isna(batch_aligned).any():
            raise ValueError("Some samples in expression matrix are missing batch labels.")
        combat = pycombat.Combat()
        expr_corrected = combat.fit_transform(expr.values, batch_aligned)
        df = pd.DataFrame(expr_corrected, index=expr.index, columns=expr.columns).T
        expr_after_batch = df.copy()
    else:
        expr_after_batch = None
    # =========================
    # 对 before / after / final 使用同样的 log 规则
    # =========================
    # print(microarray_logged)
    if data_type == "RNA-seq":
        expr_before_batch = expr_before_batch.clip(lower=0)
        expr_before_batch = np.log1p(expr_before_batch)
        if expr_after_batch is not None:
            expr_after_batch = expr_after_batch.clip(lower=0)
            expr_after_batch = np.log1p(expr_after_batch)
            # print("ceshi1")
        df = df.clip(lower=0)
        df = np.log1p(df)
    elif data_type == "microarray" and microarray_logged is False:
        expr_before_batch = expr_before_batch.clip(lower=0)
        expr_before_batch = np.log1p(expr_before_batch)
        if expr_after_batch is not None:
            expr_after_batch = expr_after_batch.clip(lower=0)
            expr_after_batch = np.log1p(expr_after_batch)
            # print("ceshi2")
        df = df.clip(lower=0)
        df = np.log1p(df)
    expr_preprocessed = df.groupby(df.index).mean()
    # print("ceshi3")
    expr_before_batch = expr_before_batch.groupby(expr_before_batch.index).mean()
    if expr_after_batch is not None:
        expr_after_batch = expr_after_batch.groupby(expr_after_batch.index).mean()

    return {
        "expr_preprocessed": expr_preprocessed,
        "expr_before_batch": expr_before_batch,
        "expr_after_batch": expr_after_batch,
    }

