from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from stableggm.edge_selection import select_edges


def edge_table_to_matrix(
    edge_df: pd.DataFrame,
    value_col: str = "norm_pcor"
) -> pd.DataFrame:
    """
    将 edge table 转回对称矩阵。
    要求 edge_df 至少包含:
    - gene1
    - gene2
    - value_col
    """
    required_cols = {"gene1", "gene2", value_col}
    missing = required_cols - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge_df is missing required columns: {missing}")

    genes = sorted(set(edge_df["gene1"].astype(str)) | set(edge_df["gene2"].astype(str)))
    matrix = pd.DataFrame(0.0, index=genes, columns=genes)

    for _, row in edge_df.iterrows():
        g1 = str(row["gene1"])
        g2 = str(row["gene2"])
        val = float(row[value_col])

        matrix.loc[g1, g2] = val
        matrix.loc[g2, g1] = val

    np.fill_diagonal(matrix.values, 0.0)
    return matrix

def test_select_edges_after_normalization():
    # 1. 输入文件：已经标准化好的边表
    input_path = Path("./normalization_outputs/aggregated_edge_df_normalized.csv")

    # 2. 输出目录
    output_dir = Path("./select_edges_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = output_dir / "normalized_matrix.csv"
    selected_csv = output_dir / "selected_edges_genenet_like.csv"

    # 3. 读取已经标准化好的边表
    norm_df = pd.read_csv(input_path)

    # 4. 基本检查
    required_cols = {"gene1", "gene2", "norm_pcor"}
    missing = required_cols - set(norm_df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    # 5. 转成对称矩阵
    matrix_df = edge_table_to_matrix(
        norm_df,
        value_col="norm_pcor"
    )
    matrix_df.to_csv(matrix_csv)

    # 6. 选边（GeneNet-like）
    net_df = select_edges(
        matrix_df=matrix_df,
        method="python_genenet_like",
        prob_threshold=0.9,
        fdr_alpha=0.1,
        random_state=42
    )
    net_df.to_csv(selected_csv, index=False)

    # 7. 输出信息
    print("Normalized edge_df shape:", norm_df.shape)
    print("Matrix shape:", matrix_df.shape)
    print("Selected edges shape:", net_df.shape)

    if not net_df.empty:
        print("\nTop selected edges:")
        print(net_df.head(10))

    print("\nSaved files:")
    print(" -", matrix_csv)
    print(" -", selected_csv)

    # 8. 简单断言
    assert matrix_csv.exists()
    assert selected_csv.exists()
    assert "norm_pcor" in norm_df.columns
    assert isinstance(net_df, pd.DataFrame)

    # 如果有结果，检查关键列是否存在
    if not net_df.empty:
        expected_cols = {"gene1", "gene2", "weight", "zscore", "pvalue", "qvalue", "lfdr", "prob"}
        missing_out = expected_cols - set(net_df.columns)
        assert not missing_out, f"Selected edge table missing columns: {missing_out}"


if __name__ == "__main__":
    test_select_edges_after_normalization()