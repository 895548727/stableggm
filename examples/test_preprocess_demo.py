from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from stableggm.preprocess import preprocess_expression
from stableggm.plotting import plot_batch_correction_boxplots


def test_preprocess_with_batch_boxplot():
    expr_path = Path("../tests/data/gene_expression_Acinetobacter_baumannii.csv")
    batch_path = Path("../tests/ab_batch_expanded_clean.csv")

    # 1. 读取表达矩阵
    expr_df = pd.read_csv(expr_path, index_col=0)
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
    expr_df = expr_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    print(expr_df)
    # 2. 读取真实批次信息
    batch_df = pd.read_csv(batch_path)
    batch_series = pd.Series(batch_df["batch"].values, index=batch_df["sample"].values)

    # 3. 对齐批次顺序到表达矩阵样本顺序
    batch_series = batch_series.reindex(expr_df.columns)
    print(batch_series)
    # 4. 预处理
    expr_after = preprocess_expression(
        expr_df=expr_df,
        data_type="microarray",
        normalization=None,
        batch_series=batch_series,
        zero_threshold=0.5,
        microarray_logged=False
    )

    # 5. 为绘图对齐前后矩阵
    expr_before_for_plot = expr_df.loc[expr_after.index, expr_after.columns]
    print(expr_before_for_plot)
    print(expr_after)
    # 6. 画 preprocess 前后箱线图
    save_path = Path("./preprocess_boxplot.png")
    plot_batch_correction_boxplots(
        expr_before=expr_before_for_plot,
        expr_after=expr_after,
        batch_series=batch_series,
        title_before="Before preprocess",
        title_after="After preprocess",
        save_path=str(save_path),
        show=False
    )

    assert expr_after.shape[0] > 0
    assert expr_after.shape[1] == expr_df.shape[1]
    assert save_path.exists()
test_preprocess_with_batch_boxplot()