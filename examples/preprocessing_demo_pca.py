import pandas as pd
import numpy as np
import pycombat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from stableggm.plotting import plot_batch_correction_boxplots

# 1. 读取 TPM 数据
expr_df = pd.read_csv("../tests/data/gene_expression_Acinetobacter_baumannii.csv", index_col=0)
expr_df = expr_df.apply(pd.to_numeric, errors="coerce")
expr_df = expr_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

# 2. 读取批次信息
batch_df = pd.read_csv("../tests/ab_batch_expanded_clean.csv")
batch_series = pd.Series(batch_df["batch"].values, index=batch_df["sample"].values)
batch_series = batch_series.reindex(expr_df.columns)

# 检查批次信息是否缺失
if batch_series.isna().any():
    missing_samples = batch_series[batch_series.isna()].index.tolist()
    raise ValueError(f"Missing batch labels for samples: {missing_samples}")

# 3. 过滤高零值基因
zero_threshold = 0.5
mask = (expr_df == 0).sum(axis=1) <= (zero_threshold * expr_df.shape[1])
expr_filtered = expr_df.loc[mask].copy()

# 4. TPM 数据先 log1p
expr_before = np.log1p(expr_filtered)

# 5. ComBat 去批次
combat = pycombat.Combat()
expr_corrected = combat.fit_transform(expr_before.T.values, batch_series.values)
expr_after = pd.DataFrame(expr_corrected, index=expr_before.columns, columns=expr_before.index).T
print(expr_after)
# 6. 画 pycombat 前后箱线图
plot_batch_correction_boxplots(
    expr_before=expr_before,
    expr_after=expr_after,
    batch_series=batch_series,
    title_before="Before ComBat",
    title_after="After ComBat",
    save_path="./combat_boxplot.png",
    show=False
)
# 7. PCA before/after，按 batch 着色
# PCA 输入格式：行=样本，列=基因，所以要转置
X_before = expr_before.T
X_after = expr_after.T

pca_before = PCA(n_components=2)
pcs_before = pca_before.fit_transform(X_before)

pca_after = PCA(n_components=2)
pcs_after = pca_after.fit_transform(X_after)

plot_df_before = pd.DataFrame({
    "PC1": pcs_before[:, 0],
    "PC2": pcs_before[:, 1],
    "batch": batch_series.loc[X_before.index].values
}, index=X_before.index)

plot_df_after = pd.DataFrame({
    "PC1": pcs_after[:, 0],
    "PC2": pcs_after[:, 1],
    "batch": batch_series.loc[X_after.index].values
}, index=X_after.index)

batches = list(pd.unique(batch_series.values))
cmap = plt.get_cmap("tab20")
color_map = {b: cmap(i % 20) for i, b in enumerate(batches)}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

for b in batches:
    sub = plot_df_before[plot_df_before["batch"] == b]
    axes[0].scatter(
        sub["PC1"], sub["PC2"],
        s=28, alpha=0.75,
        color=color_map[b],
        edgecolors="none",
        label=str(b)
    )
axes[0].set_title(
    f"Before ComBat\nPC1={pca_before.explained_variance_ratio_[0]*100:.1f}%, "
    f"PC2={pca_before.explained_variance_ratio_[1]*100:.1f}%"
)
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
for b in batches:
    sub = plot_df_after[plot_df_after["batch"] == b]
    axes[1].scatter(
        sub["PC1"], sub["PC2"],
        s=28, alpha=0.75,
        color=color_map[b],
        edgecolors="none",
        label=str(b)
    )
axes[1].set_title(
    f"After ComBat\nPC1={pca_after.explained_variance_ratio_[0]*100:.1f}%, "
    f"PC2={pca_after.explained_variance_ratio_[1]*100:.1f}%"
)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
# 只放一个总图例，放到图外
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Batch",
    loc="center left",
    bbox_to_anchor=(0.86, 0.5),
    frameon=False,
    fontsize=8,
    title_fontsize=9,
    ncol=1
)

plt.subplots_adjust(right=0.84, wspace=0.28)
plt.savefig("./combat_pca_pretty.png", bbox_inches="tight")
plt.close()