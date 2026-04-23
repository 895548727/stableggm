from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 测试环境不弹窗

from stableggm.normalization import normalize_pcor_column
from stableggm.plotting import plot_normalization_distributions


def test_normalization_from_aggregated_edge_df():
    # 1. 输入文件
    input_path = Path("./aggregated_edge_df.csv")

    # 2. 输出文件
    output_dir = Path("./normalization_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "aggregated_edge_df_normalized.csv"
    hist_png = output_dir / "pcor_hist_before_after.png"
    density_png = output_dir / "pcor_density_before_after.png"

    # 3. 读取聚合边表
    edge_df = pd.read_csv(input_path)

    # 4. 调用 normalization.py
    norm_df = normalize_pcor_column(
        edge_df=edge_df,
        pcor_col="pcor"
    )

    # 5. 保存标准化结果
    norm_df.to_csv(output_csv, index=False)

    # 6. 画标准化前后分布图
    plot_normalization_distributions(
        edge_df=norm_df,
        pcor_col="pcor",
        norm_col="norm_pcor",
        hist_path=str(hist_png),
        density_path=str(density_png),
        show=False
    )

    # 7. 简单检查
    assert output_csv.exists()
    assert hist_png.exists()
    assert density_png.exists()
    assert "z_pcor" in norm_df.columns
    assert "norm_pcor" in norm_df.columns

    print("Normalization finished.")
    print("Saved normalized table:", output_csv)
    print("Saved histogram:", hist_png)
    print("Saved density plot:", density_png)


if __name__ == "__main__":
    test_normalization_from_aggregated_edge_df()