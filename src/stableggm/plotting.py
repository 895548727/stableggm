from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .network import extract_largest_component, get_degree_table, get_weighted_degree_table
from matplotlib.patches import Patch
try:
    from matplotlib_venn import venn2, venn3
except ImportError:
    venn2 = None
    venn3 = None
# =========================================================
# 基础工具函数
# =========================================================
def _get_distinct_colors(n: int, cmap_name: str = "tab20") -> list:
    """
    为柱状图等离散对象生成尽量不同的颜色。
    """
    if n <= 0:
        return []
    cmap = plt.get_cmap(cmap_name)
    if n <= 20:
        return [cmap(i) for i in range(n)]
    # n 很大时均匀采样
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _get_bar_colors(n: int, cmap_name: str = "tab20") -> list:
    """
    专门给柱状图用的颜色列表。
    """
    return _get_distinct_colors(n=n, cmap_name=cmap_name)

def _get_layout(
    G: nx.Graph,
    layout: str = "spring",
    seed: int = 42
):
    """
    根据 layout 名称生成节点坐标。
    """
    if layout == "spring":
        return nx.spring_layout(G, seed=seed)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "shell":
        return nx.shell_layout(G)
    raise ValueError("layout must be one of: spring, kamada_kawai, circular, shell")

def _get_edge_widths(
    G: nx.Graph,
    weight_attr: str = "weight",
    base_width: float = 1.0,
    scale: float = 3.0
) -> list[float]:
    """
    根据边权生成绘图线宽，默认使用 |weight| 控制粗细。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    widths = []
    for _, _, data in G.edges(data=True):
        w = abs(float(data.get(weight_attr, 1.0)))
        widths.append(base_width + scale * w)
    return widths

def _save_or_show(save_path: str | None = None, show: bool = True):
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def _get_module_color_map(membership_df: pd.DataFrame):
    """
    为每个 module_id 分配一个颜色。
    """
    if not isinstance(membership_df, pd.DataFrame):
        raise TypeError("membership_df must be a pandas DataFrame.")
    required_cols = {"gene", "module_id"}
    missing = required_cols - set(membership_df.columns)
    if missing:
        raise ValueError(f"membership_df is missing required columns: {missing}")
    modules = sorted(membership_df["module_id"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    color_map = {}
    for i, module_id in enumerate(modules):
        color_map[module_id] = cmap(i % 20)
    return color_map

def _get_node_colors_by_module(
    G: nx.Graph,
    membership_df: pd.DataFrame,
    default_color: str = "lightgray"
):
    """
    根据模块给节点分配颜色。
    """
    gene_to_module = dict(zip(membership_df["gene"], membership_df["module_id"]))
    module_color_map = _get_module_color_map(membership_df)
    node_colors = []
    for node in G.nodes():
        module_id = gene_to_module.get(node, None)
        if module_id is None:
            node_colors.append(default_color)
        else:
            node_colors.append(module_color_map.get(module_id, default_color))
    return node_colors

# =========================================================
# 1. normalization / batch correction 可视化
# =========================================================
def plot_normalization_distributions(
    edge_df: pd.DataFrame,
    pcor_col: str = "pcor",
    norm_col: str = "norm_pcor",
    hist_path: str | None = None,
    density_path: str | None = None,
    show: bool = True
):
    """
    绘制标准化前后的直方图和密度图。
    """
    if not isinstance(edge_df, pd.DataFrame):
        raise TypeError("edge_df must be a pandas DataFrame.")
    if pcor_col not in edge_df.columns:
        raise ValueError(f"Column '{pcor_col}' not found in edge_df.")
    if norm_col not in edge_df.columns:
        raise ValueError(f"Column '{norm_col}' not found in edge_df.")
    # Histogram
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].hist(edge_df[pcor_col].values, bins=300)
    axes[0].set_title("Before normalization")
    axes[0].set_xlabel("ppcor")
    axes[1].hist(edge_df[norm_col].values, bins=300)
    axes[1].set_title("After normalization")
    axes[1].set_xlabel("ppcor")
    if hist_path is not None:
        plt.tight_layout()
        plt.savefig(hist_path, transparent=True, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    # Density
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    edge_df[pcor_col].plot(kind="density", ax=axes[0])
    axes[0].set_title("Before normalization")
    axes[0].set_xlabel("ppcor")
    edge_df[norm_col].plot(kind="density", ax=axes[1])
    axes[1].set_title("After normalization")
    axes[1].set_xlabel("ppcor")
    if density_path is not None:
        plt.tight_layout()
        plt.savefig(density_path, transparent=True, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_batch_correction_boxplots(
    expr_before: pd.DataFrame,
    expr_after: pd.DataFrame,
    batch_series: pd.Series | None = None,
    sample_axis: str = "columns",
    figsize: tuple[int, int] = (12, 8),
    title_before: str = "Before batch correction",
    title_after: str = "After batch correction",
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制 ComBat 去批次效应前后的箱线图。
    参数
    ----
    expr_before : pd.DataFrame
        去批次前表达矩阵，默认行=gene，列=sample
    expr_after : pd.DataFrame
        去批次后表达矩阵，默认行=gene，列=sample
    batch_series : pd.Series or None
        样本对应批次信息，index 建议与 sample 名一致
        如果提供，会在 x 轴标签中标出 batch
    sample_axis : str
        "columns" 表示列为 sample；"index" 表示行为 sample
    """
    if not isinstance(expr_before, pd.DataFrame) or not isinstance(expr_after, pd.DataFrame):
        raise TypeError("expr_before and expr_after must be pandas DataFrames.")
    if expr_before.shape[1] != expr_after.shape[1]:
        raise ValueError("expr_before and expr_after must have the same number of samples.")
    if sample_axis not in {"columns", "index"}:
        raise ValueError("sample_axis must be 'columns' or 'index'.")
    if sample_axis == "columns":
        before_plot = expr_before.copy()
        after_plot = expr_after.copy()
        sample_names = list(before_plot.columns)
    else:
        before_plot = expr_before.T.copy()
        after_plot = expr_after.T.copy()
        sample_names = list(before_plot.columns)
    labels = sample_names
    if batch_series is not None:
        batch_series = batch_series.copy()
        batch_series.index = batch_series.index.astype(str)
        labels = [
            f"{s}\n({batch_series.loc[s]})" if s in batch_series.index else str(s)
            for s in map(str, sample_names)
        ]
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].boxplot([before_plot[col].dropna().values for col in before_plot.columns], labels=labels, showfliers=False)
    axes[0].set_title(title_before)
    axes[0].set_ylabel("Expression")

    axes[1].boxplot([after_plot[col].dropna().values for col in after_plot.columns], labels=labels, showfliers=False)
    axes[1].set_title(title_after)
    axes[1].set_ylabel("Expression")
    axes[1].set_xlabel("Samples")
    # 去掉横坐标刻度和标签
    axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes[1].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    _save_or_show(save_path=save_path, show=show)

def plot_batch_correction_pca(
    expr_before: pd.DataFrame,
    expr_after: pd.DataFrame,
    batch_series: pd.Series | None = None,
    sample_axis: str = "columns",
    figsize: tuple[int, int] = (14, 5.5),
    dpi: int = 300,
    title_before: str = "Before ComBat",
    title_after: str = "After ComBat",
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制 ComBat 去批次效应前后的 PCA 散点图（无图例版）。
    """
    if not isinstance(expr_before, pd.DataFrame) or not isinstance(expr_after, pd.DataFrame):
        raise TypeError("expr_before and expr_after must be pandas DataFrames.")
    if sample_axis not in {"columns", "index"}:
        raise ValueError("sample_axis must be 'columns' or 'index'.")

    if sample_axis == "columns":
        X_before = expr_before.T.copy()
        X_after = expr_after.T.copy()
    else:
        X_before = expr_before.copy()
        X_after = expr_after.copy()

    if X_before.shape[0] != X_after.shape[0]:
        raise ValueError("expr_before and expr_after must have the same number of samples.")

    sample_names = list(map(str, X_before.index))

    if batch_series is None:
        batch_series = pd.Series(["batch1"] * len(sample_names), index=sample_names)
    else:
        batch_series = batch_series.copy()
        batch_series.index = batch_series.index.astype(str)
        missing = [s for s in sample_names if s not in batch_series.index]
        if missing:
            raise ValueError(f"batch_series is missing samples, for example: {missing[:5]}")
        batch_series = batch_series.loc[sample_names]

    # PCA
    pca_before = PCA(n_components=2)
    pcs_before = pca_before.fit_transform(X_before)

    pca_after = PCA(n_components=2)
    pcs_after = pca_after.fit_transform(X_after)

    plot_df_before = pd.DataFrame({
        "PC1": pcs_before[:, 0],
        "PC2": pcs_before[:, 1],
        "batch": batch_series.values
    }, index=sample_names)

    plot_df_after = pd.DataFrame({
        "PC1": pcs_after[:, 0],
        "PC2": pcs_after[:, 1],
        "batch": batch_series.values
    }, index=sample_names)

    batches = list(pd.unique(batch_series.values))
    cmap = plt.get_cmap("tab20")
    color_map = {b: cmap(i % 20) for i, b in enumerate(batches)}

    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"wspace": 0.32}
    )

    # Before
    for b in batches:
        sub = plot_df_before[plot_df_before["batch"] == b]
        axes[0].scatter(
            sub["PC1"],
            sub["PC2"],
            s=28,
            alpha=0.75,
            color=color_map[b],
            edgecolors="none"
        )

    axes[0].set_title(
        f"{title_before}\nPC1={pca_before.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2={pca_before.explained_variance_ratio_[1]*100:.1f}%"
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # After
    for b in batches:
        sub = plot_df_after[plot_df_after["batch"] == b]
        axes[1].scatter(
            sub["PC1"],
            sub["PC2"],
            s=28,
            alpha=0.75,
            color=color_map[b],
            edgecolors="none"
        )

    axes[1].set_title(
        f"{title_after}\nPC1={pca_after.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2={pca_after.explained_variance_ratio_[1]*100:.1f}%"
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if not show:
            plt.close()
    else:
        if show:
            plt.show()
        else:
            plt.close()
# =========================================================
# 2. 网络基础可视化
# =========================================================
def draw_graph(
    G: nx.Graph,
    with_labels: bool = True,
    node_size: int = 300,
    font_size: int = 8,
    weight_attr: str = "weight",
    figsize: tuple[int, int] = (8, 6),
    title: str | None = None,
    save_path: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    label_top_n: int | None = 20,
    show: bool = True
):
    """
    可视化整个网络。
    参数
    ----
    label_top_n : int | None
        若为 None，则给所有节点加标签；
        若为正整数，则仅给 degree 最高的前 N 个节点加标签。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    if label_top_n is not None:
        if not isinstance(label_top_n, int):
            raise TypeError("label_top_n must be an int or None.")
        if label_top_n < 1:
            raise ValueError("label_top_n must be >= 1 when provided.")
    pos = _get_layout(G, layout=layout, seed=seed)
    edge_widths = _get_edge_widths(G, weight_attr=weight_attr)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
    if with_labels:
        if label_top_n is None:
            # 所有节点都标注
            labels = {node: str(node) for node in G.nodes()}
        else:
            # 仅标注 degree 最高的前 N 个节点
            degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in degree_sorted[:label_top_n]]
            labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    _save_or_show(save_path=save_path, show=show)

def draw_largest_component(
    G: nx.Graph,
    with_labels: bool = True,
    node_size: int = 80,
    font_size: int = 7,
    weight_attr: str = "weight",
    figsize: tuple[int, int] = (12, 10),
    title: str | None = None,
    save_path: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    label_top_n: int | None = 20,
    show: bool = True
):
    """
    只画最大连通分量。
    节点大小和颜色都按 degree 映射：
    degree 越大，节点越大、颜色越深。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    if label_top_n is not None:
        if not isinstance(label_top_n, int):
            raise TypeError("label_top_n must be an int or None.")
        if label_top_n < 1:
            raise ValueError("label_top_n must be >= 1 when provided.")

    largest_G = extract_largest_component(G)
    if largest_G.number_of_nodes() == 0:
        raise ValueError("Largest connected component is empty.")
    # 布局：尽量拉开
    if layout == "spring":
        pos = nx.spring_layout(
            largest_G,
            seed=seed,
            k=2.2 / np.sqrt(max(largest_G.number_of_nodes(), 1)),
            iterations=300
        )
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(largest_G)
    elif layout == "circular":
        pos = nx.circular_layout(largest_G)
    elif layout == "shell":
        pos = nx.shell_layout(largest_G)
    else:
        raise ValueError("layout must be one of: spring, kamada_kawai, circular, shell")
    # degree
    degree_dict = dict(largest_G.degree())
    degree_values = np.array([degree_dict[n] for n in largest_G.nodes()], dtype=float)
    if len(degree_values) == 0:
        raise ValueError("No degree values found in largest component.")
    d_min = degree_values.min()
    d_max = degree_values.max()
    if d_max > d_min:
        degree_norm = (degree_values - d_min) / (d_max - d_min)
    else:
        degree_norm = np.full_like(degree_values, 0.5)
    # 节点大小：degree 越大越大
    node_sizes = node_size + 500 * degree_norm
    # 节点颜色：degree 越大越深
    cmap = plt.cm.Blues
    node_colors = [cmap(0.35 + 0.60 * x) for x in degree_norm]
    # 边尽量淡一些
    edge_widths = _get_edge_widths(
        largest_G,
        weight_attr=weight_attr,
        base_width=0.3,
        scale=1.0
    )
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(
        largest_G,
        pos,
        width=edge_widths,
        alpha=0.15,
        edge_color="black"
    )
    nx.draw_networkx_nodes(
        largest_G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.4
    )
    if with_labels:
        if label_top_n is None:
            labels = {node: str(node) for node in largest_G.nodes()}
        else:
            degree_sorted = sorted(largest_G.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in degree_sorted[:label_top_n]]
            labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(
            largest_G,
            pos,
            labels=labels,
            font_size=font_size,
            font_color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15)
        )
    if title is None:
        title = "Largest Connected Component"
    plt.title(title, fontsize=16)
    plt.axis("off")
    _save_or_show(save_path=save_path, show=show)

def draw_graph_by_degree(
    G: nx.Graph,
    with_labels: bool = True,
    font_size: int = 7,
    weight_attr: str = "weight",
    figsize: tuple[int, int] = (12, 10),
    title: str | None = None,
    save_path: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    min_node_size: int = 80,
    scale: float = 35.0,
    label_top_n: int | None = 20,
    cmap: str = "Reds",
    show: bool = True
):
    """
    按 degree 绘制整个网络：
    - 节点大小随 degree 增大
    - 节点颜色随 degree 增大而加深
    - 默认仅标注 degree 最高的前 label_top_n 个节点
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    if label_top_n is not None:
        if not isinstance(label_top_n, int):
            raise TypeError("label_top_n must be an int or None.")
        if label_top_n < 1:
            raise ValueError("label_top_n must be >= 1 when provided.")
    degree_dict = dict(G.degree())
    degrees = np.array([degree_dict[node] for node in G.nodes()], dtype=float)
    # 1. 节点大小：随 degree 增大
    node_sizes = min_node_size + scale * degrees
    # 2. 节点颜色：随 degree 增大而加深
    if degrees.max() == degrees.min():
        norm_degrees = np.ones_like(degrees) * 0.5
    else:
        norm_degrees = (degrees - degrees.min()) / (degrees.max() - degrees.min())
    node_colors = plt.get_cmap(cmap)(0.25 + 0.75 * norm_degrees)
    # 3. 布局：尽量拉开节点
    if layout == "spring":
        pos = nx.spring_layout(
            G,
            seed=seed,
            k=2.0 / np.sqrt(max(G.number_of_nodes(), 1)),
            iterations=300
        )
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        raise ValueError("layout must be one of: spring, kamada_kawai, circular, shell")
    edge_widths = _get_edge_widths(
        G,
        weight_attr=weight_attr,
        base_width=0.3,
        scale=1.0
    )
    plt.figure(figsize=figsize)
    # 4. 先画边，淡一点
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.15,
        edge_color="black"
    )
    # 5. 再画点
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.4
    )
    # 6. 标签：只标前 label_top_n 个高 degree 节点
    if with_labels:
        if label_top_n is None:
            labels = {node: str(node) for node in G.nodes()}
        else:
            degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in degree_sorted[:label_top_n]]
            labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=font_size,
            font_color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15)
        )
    # 7. 图例：给几个代表性的 degree 尺度
    if len(degrees) > 0:
        deg_min = int(degrees.min())
        deg_med = int(np.median(degrees))
        deg_max = int(degrees.max())
        legend_degrees = sorted(set([deg_min, deg_med, deg_max]))
        handles = [
            plt.scatter(
                [], [],
                s=min_node_size + scale * d,
                color=plt.get_cmap(cmap)(0.25 + 0.75 * (
                    0.5 if degrees.max() == degrees.min()
                    else (d - degrees.min()) / (degrees.max() - degrees.min())
                )),
                edgecolors="white",
                linewidths=0.4
            )
            for d in legend_degrees
        ]
        labels = [f"Degree {d}" for d in legend_degrees]

        plt.legend(
            handles=handles,
            labels=labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            title="Node degree",
            title_fontsize=10
        )

    if title is None:
        title = "Network by Degree"
    plt.title(title, fontsize=16)
    plt.axis("off")
    _save_or_show(save_path=save_path, show=show)

def plot_degree_distribution(
    G: nx.Graph,
    bins: int = 30,
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制度分布直方图。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    degrees = [d for _, d in G.degree()]
    plt.figure(figsize=figsize)
    plt.hist(degrees, bins=bins)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    if title is None:
        title = "Degree Distribution"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

def plot_degree_distribution_loglog(
    G: nx.Graph,
    k_min: int | None = None,
    fit_tail: bool = True,
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制度分布的 log-log 散点图，并可选对尾部做线性拟合。

    参数
    ----
    G : nx.Graph
        网络图
    k_min : int | None
        拟合尾部的最小 degree。若为 None，则默认用 degree >= 3 的点做拟合。
    fit_tail : bool
        是否对尾部做线性拟合
    figsize : tuple
        图尺寸
    title : str | None
        图标题
    save_path : str | None
        保存路径
    show : bool
        是否显示图片

    返回
    ----
    result : dict
        {
            "degree_values": np.ndarray,
            "degree_counts": np.ndarray,
            "log_k": np.ndarray,
            "log_Nk": np.ndarray,
            "fit_slope": float | None,
            "fit_intercept": float | None,
            "fit_r2": float | None,
            "k_min": int | None
        }
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    if len(degrees) == 0:
        raise ValueError("No degree values found.")
    # 统计 N(k)
    unique_k, counts = np.unique(degrees, return_counts=True)
    # 去掉 k=0，避免 log(0)
    mask = unique_k > 0
    unique_k = unique_k[mask]
    counts = counts[mask]
    if len(unique_k) == 0:
        raise ValueError("All nodes have degree 0; cannot draw log-log plot.")
    log_k = np.log10(unique_k)
    log_Nk = np.log10(counts)
    fit_slope = None
    fit_intercept = None
    fit_r2 = None
    plt.figure(figsize=figsize)
    # 观测点
    plt.scatter(
        log_k,
        log_Nk,
        s=28,
        facecolors="white",
        edgecolors="gray",
        linewidths=1.0,
        label="observed (log-log)"
    )
    # 尾部拟合
    if fit_tail:
        if k_min is None:
            k_min = 3
        fit_mask = unique_k >= k_min
        if fit_mask.sum() >= 2:
            x_fit = log_k[fit_mask]
            y_fit = log_Nk[fit_mask]
            fit_slope, fit_intercept = np.polyfit(x_fit, y_fit, 1)
            y_pred = fit_slope * x_fit + fit_intercept
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            fit_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
            y_line = fit_slope * x_line + fit_intercept
            plt.plot(
                x_line,
                y_line,
                color="#4C78A8",
                linewidth=1.8,
                label=f"linear fit (k ≥ {k_min})"
            )
            eq_text = (
                f"fitted region: k ≥ {k_min}\n"
                f"y = {fit_slope:.2f}x + {fit_intercept:.2f}\n"
                f"R² = {fit_r2:.3f}"
            )
            plt.text(
                0.04, 0.08,
                eq_text,
                transform=plt.gca().transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.9)
            )
    plt.xlabel("log(k)")
    plt.ylabel("log(N(k))")
    if title is None:
        title = "Degree distribution (log-log)"
    plt.title(title)
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.2)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return {
        "degree_values": unique_k,
        "degree_counts": counts,
        "log_k": log_k,
        "log_Nk": log_Nk,
        "fit_slope": fit_slope,
        "fit_intercept": fit_intercept,
        "fit_r2": fit_r2,
        "k_min": k_min if fit_tail else None,
    }
# =========================================================
# 3. 模块相关可视化
# =========================================================
def draw_graph_by_module(
    G: nx.Graph,
    membership_df: pd.DataFrame,
    with_labels: bool = True,
    node_size: int = 120,
    font_size: int = 7,
    weight_attr: str = "weight",
    figsize: tuple[int, int] = (12, 10),
    title: str | None = None,
    save_path: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    label_top_n: int | None = 20,
    top_n_modules: int = 10,
    default_color: str = "#D3D3D3",
    show: bool = True
):
    """
    按 module 着色绘制整个网络。
    仅在图例中展示前 top_n_modules 个最大模块，其余模块统一为灰色。
    默认仅标注 degree 最高的前 label_top_n 个节点。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    required_cols = {"gene", "module_id"}
    missing = required_cols - set(membership_df.columns)
    if missing:
        raise ValueError(f"membership_df is missing required columns: {missing}")
    if label_top_n is not None:
        if not isinstance(label_top_n, int):
            raise TypeError("label_top_n must be an int or None.")
        if label_top_n < 1:
            raise ValueError("label_top_n must be >= 1 when provided.")
    if not isinstance(top_n_modules, int) or top_n_modules < 1:
        raise ValueError("top_n_modules must be a positive integer.")
    membership_df = membership_df.copy()
    # 1. 找最大的前 top_n_modules 个模块
    module_sizes = (
        membership_df.groupby("module_id")["gene"]
        .count()
        .sort_values(ascending=False)
    )
    top_modules = list(module_sizes.head(top_n_modules).index)
    # 2. 只有前 top_n_modules 个模块保留彩色，其余模块统一灰色
    membership_df["module_id_plot"] = membership_df["module_id"].where(
        membership_df["module_id"].isin(top_modules),
        other="Other"
    )
    cmap = plt.get_cmap("tab10")
    module_color_map = {m: cmap(i % 10) for i, m in enumerate(top_modules)}
    module_color_map["Other"] = default_color
    gene_to_module = dict(zip(membership_df["gene"], membership_df["module_id_plot"]))
    node_colors = [
        module_color_map.get(gene_to_module.get(node, "Other"), default_color)
        for node in G.nodes()
    ]
    # 3. 布局：加大 spring layout 的 k，并增加迭代次数，让节点尽量分开
    if layout == "spring":
        pos = nx.spring_layout(
            G,
            seed=seed,
            k=2.0 / np.sqrt(max(G.number_of_nodes(), 1)),
            iterations=300
        )
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        raise ValueError("layout must be one of: spring, kamada_kawai, circular, shell")
    edge_widths = _get_edge_widths(
        G,
        weight_attr=weight_attr,
        base_width=0.3,
        scale=1.0
    )
    plt.figure(figsize=figsize)
    # 4. 先画边，淡一点
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.15,
        edge_color="black"
    )
    # 5. 再画点
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.4
    )
    # 6. 标签：只标前 label_top_n 个高 degree 节点
    if with_labels:
        if label_top_n is None:
            labels = {node: str(node) for node in G.nodes()}
        else:
            degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in degree_sorted[:label_top_n]]
            labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=font_size,
            font_color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15)
        )
    # 7. 图例：前 top_n_modules 个模块 + Other
    legend_handles = [
        Patch(facecolor=module_color_map[m], edgecolor="none", label=f"Module {m}")
        for m in top_modules
    ]
    legend_handles.append(
        Patch(facecolor=default_color, edgecolor="none", label="Other modules")
    )
    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title=f"Top {top_n_modules} modules",
        title_fontsize=10
    )
    if title is None:
        title = "Network Colored by Module"
    plt.title(title, fontsize=16)
    plt.axis("off")
    _save_or_show(save_path=save_path, show=show)

def draw_largest_component_by_module(
    G: nx.Graph,
    membership_df: pd.DataFrame,
    with_labels: bool = True,
    node_size: int = 100,
    font_size: int = 7,
    weight_attr: str = "weight",
    figsize: tuple[int, int] = (12, 10),
    title: str | None = None,
    save_path: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    label_top_n: int | None = 20,
    top_n_modules: int = 10,
    show: bool = True
):
    """
    只画最大连通分量，并按 module 着色。
    默认仅高亮前 top_n_modules 个最大模块，其余模块用灰色。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    required_cols = {"gene", "module_id"}
    missing = required_cols - set(membership_df.columns)
    if missing:
        raise ValueError(f"membership_df is missing required columns: {missing}")
    if label_top_n is not None:
        if not isinstance(label_top_n, int):
            raise TypeError("label_top_n must be an int or None.")
        if label_top_n < 1:
            raise ValueError("label_top_n must be >= 1 when provided.")
    largest_G = extract_largest_component(G)
    largest_nodes = set(largest_G.nodes())

    sub_membership = membership_df.loc[
        membership_df["gene"].isin(largest_nodes)
    ].copy()
    if sub_membership.empty:
        raise ValueError("No membership information found for largest component.")

    # 只保留前 top_n_modules 个最大模块为彩色，其余统一灰色
    module_sizes = (
        sub_membership.groupby("module_id")["gene"]
        .count()
        .sort_values(ascending=False)
    )
    top_modules = list(module_sizes.head(top_n_modules).index)
    sub_membership["module_id_plot"] = sub_membership["module_id"].where(
        sub_membership["module_id"].isin(top_modules),
        other="Other"
    )
    cmap = plt.get_cmap("tab10")
    module_color_map = {m: cmap(i % 10) for i, m in enumerate(top_modules)}
    module_color_map["Other"] = "#D3D3D3"
    gene_to_module = dict(zip(sub_membership["gene"], sub_membership["module_id_plot"]))
    node_colors = [
        module_color_map.get(gene_to_module.get(node, "Other"), "#D3D3D3")
        for node in largest_G.nodes()
    ]
    # 布局拉开
    if layout == "spring":
        pos = nx.spring_layout(
            largest_G,
            seed=seed,
            k=2.2 / np.sqrt(max(largest_G.number_of_nodes(), 1)),
            iterations=300
        )
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(largest_G)
    elif layout == "circular":
        pos = nx.circular_layout(largest_G)
    elif layout == "shell":
        pos = nx.shell_layout(largest_G)
    else:
        raise ValueError("layout must be one of: spring, kamada_kawai, circular, shell")
    edge_widths = _get_edge_widths(
        largest_G,
        weight_attr=weight_attr,
        base_width=0.3,
        scale=1.0
    )
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(
        largest_G,
        pos,
        width=edge_widths,
        alpha=0.15,
        edge_color="black"
    )
    nx.draw_networkx_nodes(
        largest_G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.4
    )
    if with_labels:
        if label_top_n is None:
            labels = {node: str(node) for node in largest_G.nodes()}
        else:
            degree_sorted = sorted(largest_G.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in degree_sorted[:label_top_n]]
            labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(
            largest_G,
            pos,
            labels=labels,
            font_size=font_size,
            font_color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15)
        )
    # 图例
    legend_handles = [
        Patch(facecolor=module_color_map[m], edgecolor="none", label=f"Module {m}")
        for m in top_modules
    ]
    legend_handles.append(
        Patch(facecolor="#D3D3D3", edgecolor="none", label="Other modules")
    )
    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title=f"Top {top_n_modules} modules",
        title_fontsize=10
    )
    if title is None:
        title = "Largest Component Colored by Module"
    plt.title(title, fontsize=16)
    plt.axis("off")
    _save_or_show(save_path=save_path, show=show)

def draw_module_subgraph(
    G: nx.Graph,
    membership_df: pd.DataFrame,
    module_id,
    with_labels: bool = True,
    node_size: int = 400,
    font_size: int = 8,
    weight_attr: str = "weight",
    figsize: tuple[int, int] = (6, 5),
    title: str | None = None,
    save_path: str | None = None,
    layout: str = "spring",
    seed: int = 42,
    label_top_n: int | None = 20,
    show: bool = True
):
    """
    绘制指定模块的子图。
    边权越大，边越粗、颜色越深。
    """
    required_cols = {"gene", "module_id"}
    missing = required_cols - set(membership_df.columns)
    if missing:
        raise ValueError(f"membership_df is missing required columns: {missing}")
    if label_top_n is not None:
        if not isinstance(label_top_n, int):
            raise TypeError("label_top_n must be an int or None.")
        if label_top_n < 1:
            raise ValueError("label_top_n must be >= 1 when provided.")
    module_genes = membership_df.loc[
        membership_df["module_id"] == module_id, "gene"
    ].tolist()
    if len(module_genes) == 0:
        raise ValueError(f"No genes found for module_id={module_id}")
    subG = G.subgraph(module_genes).copy()
    if subG.number_of_nodes() == 0:
        raise ValueError(f"Subgraph for module_id={module_id} is empty.")
    pos = _get_layout(subG, layout=layout, seed=seed)
    # 1. 提取边权
    edge_list = list(subG.edges(data=True))
    raw_weights = []
    for _, _, data in edge_list:
        w = abs(float(data.get(weight_attr, 1.0)))
        raw_weights.append(w)
    # 2. 宽度映射：权重越大越粗
    if len(raw_weights) > 0:
        w_min = min(raw_weights)
        w_max = max(raw_weights)
        if w_max > w_min:
            norm_weights = [(w - w_min) / (w_max - w_min) for w in raw_weights]
        else:
            norm_weights = [0.5 for _ in raw_weights]
        edge_widths = [1.0 + 4.0 * w for w in norm_weights]
        # 3. 颜色映射：权重越大颜色越深
        cmap = plt.cm.Greys
        # 避免太浅，从 0.35 到 0.95 映射
        edge_colors = [cmap(0.35 + 0.60 * w) for w in norm_weights]
    else:
        edge_widths = []
        edge_colors = []
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        subG,
        pos,
        node_size=node_size,
        node_color="#2C7FB8",
        edgecolors="white",
        linewidths=0.5
    )
    nx.draw_networkx_edges(
        subG,
        pos,
        edgelist=[(u, v) for u, v, _ in edge_list],
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.9
    )
    if with_labels:
        if label_top_n is None:
            labels = {node: str(node) for node in subG.nodes()}
        else:
            degree_sorted = sorted(subG.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in degree_sorted[:label_top_n]]
            labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(
            subG,
            pos,
            labels=labels,
            font_size=font_size,
            font_color="black"
        )
    if title is None:
        title = f"Module {module_id}"
    plt.title(title)
    plt.axis("off")
    _save_or_show(save_path=save_path, show=show)

def plot_module_size_distribution(
    membership_df: pd.DataFrame,
    bins: int = 20,
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制模块大小分布图。
    仅统计 module size >= 5 的模块。
    如果没有满足条件的模块，则跳过绘图。
    """
    if not isinstance(membership_df, pd.DataFrame):
        raise TypeError("membership_df must be a pandas DataFrame.")

    required_cols = {"gene", "module_id"}
    missing = required_cols - set(membership_df.columns)
    if missing:
        raise ValueError(f"membership_df is missing required columns: {missing}")

    module_size_series = membership_df.groupby("module_id")["gene"].count()

    # 过滤掉小于 5 的模块
    module_size_series = module_size_series[module_size_series >= 5]

    if module_size_series.empty:
        print("[INFO] No modules with size >= 5 were found. Skip plotting module size distribution.")
        return

    module_sizes = module_size_series.values

    plt.figure(figsize=figsize)
    plt.hist(
        module_sizes,
        bins=bins,
        color="orange",
        edgecolor="white",
        linewidth=1.0
    )
    plt.xlabel("Module Size")
    plt.ylabel("Frequency")

    if title is None:
        title = "Module Size Distribution (size >= 5)"
    plt.title(title)

    _save_or_show(save_path=save_path, show=show)

# =========================================================
# 4. 新增：边权 / 连通分量 / hub 图
# =========================================================
def plot_edge_weight_distribution(
    edge_df: pd.DataFrame,
    weight_col: str = "weight",
    bins: int = 50,
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制最终网络边权分布直方图。
    """
    if not isinstance(edge_df, pd.DataFrame):
        raise TypeError("edge_df must be a pandas DataFrame.")
    if weight_col not in edge_df.columns:
        raise ValueError(f"'{weight_col}' not found in edge_df.")
    plt.figure(figsize=figsize)
    plt.hist(edge_df[weight_col].dropna().values, bins=bins)
    plt.xlabel(weight_col)
    plt.ylabel("Frequency")
    if title is None:
        title = "Edge Weight Distribution"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

def plot_edge_weight_density(
    edge_df: pd.DataFrame,
    weight_col: str = "weight",
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制最终网络边权密度图。
    """
    if not isinstance(edge_df, pd.DataFrame):
        raise TypeError("edge_df must be a pandas DataFrame.")
    if weight_col not in edge_df.columns:
        raise ValueError(f"'{weight_col}' not found in edge_df.")
    plt.figure(figsize=figsize)
    edge_df[weight_col].dropna().plot(kind="density")
    plt.xlabel(weight_col)
    if title is None:
        title = "Edge Weight Density"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

def plot_component_size_distribution(
    G: nx.Graph,
    bins: int = 20,
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制连通分量大小分布图。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    component_sizes = [len(c) for c in nx.connected_components(G)]
    plt.figure(figsize=figsize)
    plt.hist(component_sizes, bins=bins)
    plt.xlabel("Component Size")
    plt.ylabel("Frequency")
    if title is None:
        title = "Connected Component Size Distribution"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

def plot_top_degree_genes(
    G: nx.Graph,
    top_n: int = 20,
    figsize: tuple[int, int] = (8, 5),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制度最高的 top-N 基因条形图。
    """
    degree_df = get_degree_table(G).head(top_n)
    colors = _get_bar_colors(len(degree_df))
    plt.figure(figsize=figsize)
    plt.barh(degree_df["gene"], degree_df["degree"], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Degree")
    plt.ylabel("Gene")
    if title is None:
        title = f"Top {top_n} Hub Genes by Degree"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

def plot_top_weighted_degree_genes(
    G: nx.Graph,
    top_n: int = 20,
    figsize: tuple[int, int] = (8, 5),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制加权度最高的 top-N 基因条形图。
    """
    degree_df = get_weighted_degree_table(G).head(top_n)
    colors = _get_bar_colors(len(degree_df))
    plt.figure(figsize=figsize)
    plt.barh(degree_df["gene"], degree_df["weighted_degree"], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Weighted Degree")
    plt.ylabel("Gene")
    if title is None:
        title = f"Top {top_n} Hub Genes by Weighted Degree"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

# =========================================================
# 5. 新增：stability 诊断图
# =========================================================
def plot_presence_distribution_combined(
    stability_table: pd.DataFrame,
    n_channels: int,
    figsize: tuple[int, int] = (7, 4.8),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    融合绘制 presence_count 和 presence_ratio 分布图。

    下方横轴为 presence_count，
    上方横轴为对应的 presence_ratio，
    纵轴为 frequency。
    """
    if not isinstance(stability_table, pd.DataFrame):
        raise TypeError("stability_table must be a pandas DataFrame.")
    if "presence_count" not in stability_table.columns:
        raise ValueError("'presence_count' not found in stability_table.")
    if not isinstance(n_channels, int) or n_channels < 1:
        raise ValueError("n_channels must be a positive integer.")
    counts = (
        stability_table["presence_count"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    if counts.empty:
        raise ValueError("No valid presence_count values found.")
    x = counts.index.to_list()
    y = counts.values
    fig, ax = plt.subplots(figsize=figsize)
    # 主柱状图
    bars = ax.bar(
        x,
        y,
        width=0.55,
        color="#F4A261",      # 柔和橘色
        edgecolor="white",
        linewidth=1.2
    )
    # 给柱子上方加数值
    for bar, val in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=9
        )
    ax.set_xlabel("Presence Count", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=10)
    # 上方第二横轴：presence_ratio
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{i / n_channels:.2f}" for i in x], fontsize=10)
    ax_top.set_xlabel("Presence Ratio", fontsize=11)
    # 美化
    ax.spines["top"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    if title is None:
        title = "Presence Stability Distribution"
    ax.set_title(title, fontsize=16, pad=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_top_stable_edges(
    stability_table: pd.DataFrame,
    top_n: int = 20,
    score_col: str = "presence_count",
    figsize: tuple[int, int] = (9, 5),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制最稳定边的条形图。
    默认按 presence_count 排序。
    """
    if not isinstance(stability_table, pd.DataFrame):
        raise TypeError("stability_table must be a pandas DataFrame.")
    required_cols = {"gene1", "gene2", score_col}
    missing = required_cols - set(stability_table.columns)
    if missing:
        raise ValueError(f"stability_table is missing required columns: {missing}")
    df = stability_table.sort_values(by=score_col, ascending=False).head(top_n).copy()
    df["edge"] = df["gene1"].astype(str) + " -- " + df["gene2"].astype(str)
    colors = _get_bar_colors(len(df))

    plt.figure(figsize=figsize)
    plt.barh(df["edge"], df[score_col], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel(score_col)
    plt.ylabel("Edge")
    if title is None:
        title = f"Top {top_n} Stable Edges"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)
# =========================================================
# 6. 新增：富集分析结果图
# =========================================================
def plot_enrichment_bar(
    enrich_df: pd.DataFrame,
    top_n: int = 10,
    term_col: str = "term",
    score_col: str = "score",
    figsize: tuple[int, int] = (8, 5),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    富集结果条形图。
    默认使用 score 列（例如 -log10(qvalue)）。
    """
    if not isinstance(enrich_df, pd.DataFrame):
        raise TypeError("enrich_df must be a pandas DataFrame.")
    required_cols = {term_col, score_col}
    missing = required_cols - set(enrich_df.columns)
    if missing:
        raise ValueError(f"enrich_df is missing required columns: {missing}")

    df = enrich_df.sort_values(by=score_col, ascending=False).head(top_n).copy()
    colors = _get_bar_colors(len(df))

    plt.figure(figsize=figsize)
    plt.barh(df[term_col], df[score_col], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel(score_col)
    plt.ylabel(term_col)
    if title is None:
        title = f"Top {top_n} Enriched Terms"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

def plot_enrichment_dot(
    enrich_df: pd.DataFrame,
    top_n: int = 10,
    term_col: str = "term",
    score_col: str = "score",
    size_col: str = "overlap_size",
    figsize: tuple[int, int] = (8, 5),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    富集结果气泡图。
    x 轴为 score，点大小为 overlap_size。
    """
    if not isinstance(enrich_df, pd.DataFrame):
        raise TypeError("enrich_df must be a pandas DataFrame.")
    required_cols = {term_col, score_col, size_col}
    missing = required_cols - set(enrich_df.columns)
    if missing:
        raise ValueError(f"enrich_df is missing required columns: {missing}")
    df = enrich_df.sort_values(by=score_col, ascending=False).head(top_n).copy()
    y_positions = list(range(len(df)))
    plt.figure(figsize=figsize)
    plt.scatter(
        df[score_col],
        y_positions,
        s=df[size_col] * 30,
        alpha=0.7
    )
    plt.yticks(y_positions, df[term_col])
    plt.xlabel(score_col)
    plt.ylabel(term_col)
    if title is None:
        title = f"Top {top_n} Enriched Terms"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

def plot_enrichment_bubble(
    enrich_df: pd.DataFrame,
    top_n: int = 20,
    term_col: str = "term",
    x_col: str = "rich_factor",
    size_col: str = "gene_count",
    color_col: str = "qvalue",
    figsize: tuple[int, int] = (8, 7),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    cmap: str = "rainbow_r",
    size_scale: float = 25.0
):
    """
    富集分析气泡图
    - x轴: rich factor
    - y轴: term
    - 点大小: gene count
    - 点颜色: qvalue
    """
    if not isinstance(enrich_df, pd.DataFrame):
        raise TypeError("enrich_df must be a pandas DataFrame.")
    required_cols = {term_col, x_col, size_col, color_col}
    missing = required_cols - set(enrich_df.columns)
    if missing:
        raise ValueError(f"enrich_df is missing required columns: {missing}")
    df = enrich_df.dropna(subset=[term_col, x_col, size_col, color_col]).copy()
    if df.empty:
        raise ValueError("No valid rows available for plotting.")
    # 先按显著性和富集强度排序，再取前 top_n
    df = df.sort_values(by=[color_col, x_col], ascending=[True, False]).head(top_n).copy()
    df = df.iloc[::-1].copy()   # 最显著的放上面
    y_positions = np.arange(len(df))
    sizes = df[size_col].astype(float).values * size_scale
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        df[x_col],
        y_positions,
        s=sizes,
        c=df[color_col],
        cmap=cmap,
        alpha=0.95,
        edgecolors="none"
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df[term_col])
    ax.set_xlabel("Rich factor")
    ax.set_ylabel("")
    if title is None:
        title = "Statistics of Pathway Enrichment"
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    # colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("qvalue")
    # 气泡大小图例
    size_vals = df[size_col].astype(float)
    legend_sizes = sorted(set([
        int(size_vals.min()),
        int(np.median(size_vals)),
        int(size_vals.max())
    ]))
    handles = [
        plt.scatter([], [], s=s * size_scale, color="black")
        for s in legend_sizes
    ]
    labels = [str(s) for s in legend_sizes]
    ax.legend(
        handles,
        labels,
        title="Gene_number",
        frameon=False,
        bbox_to_anchor=(1.20, 0.35),
        loc="center left"
    )
    plt.tight_layout()
    _save_or_show(save_path=save_path, show=show)
# =========================================================
# 7. 可选：MCL inflation 敏感性图
# =========================================================
def plot_mcl_inflation_sensitivity(
    result_df: pd.DataFrame,
    inflation_col: str = "inflation",
    module_count_col: str = "n_modules",
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True
):
    """
    绘制 inflation 参数敏感性图。
    result_df 至少需要：
    - inflation
    - n_modules
    """
    if not isinstance(result_df, pd.DataFrame):
        raise TypeError("result_df must be a pandas DataFrame.")
    required_cols = {inflation_col, module_count_col}
    missing = required_cols - set(result_df.columns)
    if missing:
        raise ValueError(f"result_df is missing required columns: {missing}")
    df = result_df.sort_values(by=inflation_col)
    plt.figure(figsize=figsize)
    plt.plot(df[inflation_col], df[module_count_col], marker="o")
    plt.xlabel(inflation_col)
    plt.ylabel(module_count_col)
    if title is None:
        title = "MCL Inflation Sensitivity"
    plt.title(title)
    _save_or_show(save_path=save_path, show=show)

def plot_edges_venn(
    edge_sets: dict,
    labels: dict = None,
    figsize: tuple = (8, 7),
    title: str = "Edge Overlap across Channels",
    save_path: str = None,
    show: bool = True
):
    """
    绘制多个边集的韦恩图/重叠图。
    支持：
    - 2 个集合：venn2
    - 3 个集合：venn3
    - 4 个集合：绘制“四集合重叠摘要图”
    - >4 个集合：不绘制，直接返回
    """
    if not isinstance(edge_sets, dict):
        raise TypeError("edge_sets must be a dict")
    n = len(edge_sets)
    if n < 2:
        raise ValueError("At least 2 edge sets are required")
    if n > 4:
        print(f"[INFO] plot_edges_venn skipped: {n} edge sets provided (>4 not supported).")
        return
    if n in {2, 3} and (venn2 is None or venn3 is None):
        raise ImportError(
            "matplotlib-venn is required for Venn plotting. "
            "Install it with `pip install matplotlib-venn`."
        )
    def normalize_set(edges):
        normalized = set()
        for e in edges:
            if isinstance(e, frozenset):
                normalized.add(e)
            else:
                g1, g2 = e
                normalized.add(frozenset((g1, g2)))
        return normalized
    sets = {k: normalize_set(v) for k, v in edge_sets.items()}
    keys = list(sets.keys())
    set_list = [sets[k] for k in keys]
    set_labels = keys if labels is None else [labels.get(k, k) for k in keys]
    # 更柔和、区分度更好的颜色
    venn_colors = ["#E76F51", "#2A9D8F", "#5B5FEE", "#E9C46A"]
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    if n == 2:
        v = venn2(subsets=set_list, set_labels=set_labels, ax=ax)
        # 区域样式
        subset_ids = ["10", "01", "11"]
        for i, sid in enumerate(subset_ids):
            patch = v.get_patch_by_id(sid)
            if patch is not None:
                patch.set_color(venn_colors[i if i < 2 else 2])
                patch.set_alpha(0.45)
                patch.set_edgecolor("white")
                patch.set_linewidth(2.0)
        # 数字样式
        for sid in subset_ids:
            text = v.get_label_by_id(sid)
            if text is not None:
                text.set_fontsize(16)
                text.set_fontweight("bold")
                text.set_color("black")
        # 集合名样式
        if v.set_labels is not None:
            for t in v.set_labels:
                if t is not None:
                    t.set_fontsize(18)
                    t.set_fontweight("bold")
        ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
        ax.axis("off")
        _save_or_show(save_path, show)
        return
    if n == 3:
        v = venn3(subsets=set_list, set_labels=set_labels, ax=ax)
        # 给 7 个区域分配更协调的颜色
        patch_color_map = {
            "100": venn_colors[0],
            "010": venn_colors[1],
            "001": venn_colors[2],
            "110": venn_colors[0],
            "101": venn_colors[2],
            "011": venn_colors[1],
            "111": "#9C89B8",   # 中心交集用偏灰紫，更柔和
        }
        for sid, color in patch_color_map.items():
            patch = v.get_patch_by_id(sid)
            if patch is not None:
                patch.set_color(color)
                patch.set_alpha(0.42 if sid != "111" else 0.50)
                patch.set_edgecolor("white")
                patch.set_linewidth(2.0)
        # 数字样式
        for sid in ["100", "010", "001", "110", "101", "011", "111"]:
            text = v.get_label_by_id(sid)
            if text is not None:
                text.set_fontsize(16)
                text.set_fontweight("bold")
                text.set_color("black")
        # 集合标签样式
        if v.set_labels is not None:
            for t in v.set_labels:
                if t is not None:
                    t.set_fontsize(18)
                    t.set_fontweight("bold")
        ax.set_title(title, fontsize=20, fontweight="bold", pad=22)
        ax.axis("off")
        _save_or_show(save_path, show)
        return
    # -------------------------
    # 4 sets: 更整洁的摘要图
    # -------------------------
    if n == 4:
        ax.axis("off")
        set_a, set_b, set_c, set_d = set_list
        label_a, label_b, label_c, label_d = set_labels
        sizes = {
            label_a: len(set_a),
            label_b: len(set_b),
            label_c: len(set_c),
            label_d: len(set_d),
        }
        all_intersection = set_a & set_b & set_c & set_d
        pairwise = {
            f"{label_a} ∩ {label_b}": len(set_a & set_b),
            f"{label_a} ∩ {label_c}": len(set_a & set_c),
            f"{label_a} ∩ {label_d}": len(set_a & set_d),
            f"{label_b} ∩ {label_c}": len(set_b & set_c),
            f"{label_b} ∩ {label_d}": len(set_b & set_d),
            f"{label_c} ∩ {label_d}": len(set_c & set_d),
        }
        triple = {
            f"{label_a} ∩ {label_b} ∩ {label_c}": len(set_a & set_b & set_c),
            f"{label_a} ∩ {label_b} ∩ {label_d}": len(set_a & set_b & set_d),
            f"{label_a} ∩ {label_c} ∩ {label_d}": len(set_a & set_c & set_d),
            f"{label_b} ∩ {label_c} ∩ {label_d}": len(set_b & set_c & set_d),
        }
        colors = venn_colors[:4]
        circle_positions = [
            (0.34, 0.64, label_a, colors[0], sizes[label_a]),
            (0.66, 0.64, label_b, colors[1], sizes[label_b]),
            (0.34, 0.36, label_c, colors[2], sizes[label_c]),
            (0.66, 0.36, label_d, colors[3], sizes[label_d]),
        ]
        for x, y, lab, color, size in circle_positions:
            circle = plt.Circle((x, y), 0.17, color=color, alpha=0.35, ec="white", lw=2.0)
            ax.add_patch(circle)
            ax.text(x, y + 0.22, lab, ha="center", va="center",
                    fontsize=15, fontweight="bold")
            ax.text(x, y, f"n={size}", ha="center", va="center",
                    fontsize=14, fontweight="bold")
        ax.text(
            0.50, 0.50,
            f"4-way overlap\nn={len(all_intersection)}",
            ha="center", va="center",
            fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)
        )
        text_lines = ["Pairwise overlaps"]
        for k, v in pairwise.items():
            text_lines.append(f"{k}: {v}")
        text_lines.append("")
        text_lines.append("Triple overlaps")
        for k, v in triple.items():
            text_lines.append(f"{k}: {v}")
        ax.text(
            1.03, 0.52,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="lightgray", alpha=0.95)
        )
        legend_handles = [
            Patch(facecolor=colors[i], edgecolor="white", alpha=0.45, label=set_labels[i])
            for i in range(4)
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(1.03, 0.02),
            frameon=False,
            fontsize=11
        )
        ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
        _save_or_show(save_path, show)
