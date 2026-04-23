from __future__ import annotations

import pandas as pd
import networkx as nx

def _validate_edge_df(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    检查输入边表是否合法。
    至少需要 gene1, gene2 两列。
    额外处理：
    - 去掉 self-loop
    - 检查重复边（无向边意义下）
    - 尽量把常见数值列转为数值型
    """
    if not isinstance(edge_df, pd.DataFrame):
        raise TypeError("edge_df must be a pandas DataFrame.")
    if edge_df.empty:
        raise ValueError("edge_df is empty.")
    required_cols = {"gene1", "gene2"}
    missing = required_cols - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge_df is missing required columns: {missing}")
    df = edge_df.copy()
    df["gene1"] = df["gene1"].astype(str)
    df["gene2"] = df["gene2"].astype(str)
    # 去掉 self-loop
    df = df.loc[df["gene1"] != df["gene2"]].copy()
    if df.empty:
        raise ValueError("edge_df contains only self-loops after filtering.")
    # 尝试标准化常见数值列
    numeric_cols = ["weight", "presence_count", "presence_ratio", "score_mean", "qvalue_median"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 检查无向边重复
    edge_keys = df.apply(
        lambda row: tuple(sorted((row["gene1"], row["gene2"]))),
        axis=1
    )
    if edge_keys.duplicated().any():
        duplicated_edges = edge_keys[edge_keys.duplicated()].unique()
        raise ValueError(
            f"edge_df contains duplicated undirected edges, e.g. {list(duplicated_edges[:5])}"
        )
    return df.reset_index(drop=True)

def build_graph_from_edges(
    edge_df: pd.DataFrame,
    weight_col: str = "weight"
) -> nx.Graph:
    """
    从边表构建无向图。
    参数
    ----
    edge_df : pd.DataFrame
        至少包含 gene1, gene2
        其余列会自动作为边属性加入图中
    weight_col : str
        若存在该列，则作为 nx 的标准 'weight' 边属性
    返回
    ----
    G : nx.Graph
        networkx 无向图
    """
    edge_df = _validate_edge_df(edge_df)
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        gene1 = row["gene1"]
        gene2 = row["gene2"]
        attrs = row.to_dict()
        attrs.pop("gene1", None)
        attrs.pop("gene2", None)
        # 如果指定列存在，则保证标准化成 weight 属性
        if weight_col in row.index and pd.notna(row[weight_col]):
            attrs["weight"] = float(row[weight_col])
        G.add_edge(gene1, gene2, **attrs)
    return G

def get_degree_table(G: nx.Graph) -> pd.DataFrame:
    """
    获取节点 degree 表。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    rows = []
    for node, degree in G.degree():
        rows.append({
            "gene": node,
            "degree": degree
        })
    degree_df = pd.DataFrame(rows).sort_values(
        by=["degree", "gene"],
        ascending=[False, True]
    ).reset_index(drop=True)
    return degree_df

def get_weighted_degree_table(G: nx.Graph, weight: str = "weight") -> pd.DataFrame:
    """
    获取节点加权 degree（signed strength）。
    若边权有正有负，可能出现相互抵消。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    rows = []
    for node in G.nodes():
        strength = 0.0
        for _, _, data in G.edges(node, data=True):
            strength += float(data.get(weight, 1.0))
        rows.append({
            "gene": node,
            "weighted_degree": strength
        })
    degree_df = pd.DataFrame(rows).sort_values(
        by=["weighted_degree", "gene"],
        ascending=[False, True]
    ).reset_index(drop=True)
    return degree_df

def get_abs_weighted_degree_table(G: nx.Graph, weight: str = "weight") -> pd.DataFrame:
    """
    获取节点绝对值加权 degree。
    适合正负边同时存在的情况。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    rows = []
    for node in G.nodes():
        strength = 0.0
        for _, _, data in G.edges(node, data=True):
            strength += abs(float(data.get(weight, 1.0)))
        rows.append({
            "gene": node,
            "abs_weighted_degree": strength
        })
    degree_df = pd.DataFrame(rows).sort_values(
        by=["abs_weighted_degree", "gene"],
        ascending=[False, True]
    ).reset_index(drop=True)
    return degree_df

def get_node_table(G: nx.Graph) -> pd.DataFrame:
    """
    导出节点表。
    当前至少包含 gene 和 degree。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    rows = []
    for node in G.nodes():
        rows.append({
            "gene": node,
            "degree": G.degree(node)
        })
    node_df = pd.DataFrame(rows).sort_values(
        by=["degree", "gene"],
        ascending=[False, True]
    ).reset_index(drop=True)
    return node_df

def get_edge_table(G: nx.Graph) -> pd.DataFrame:
    """
    导出图中的边表（带全部边属性）。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    rows = []
    for gene1, gene2, data in G.edges(data=True):
        row = {
            "gene1": gene1,
            "gene2": gene2,
        }
        row.update(data)
        rows.append(row)
    if len(rows) == 0:
        return pd.DataFrame(columns=["gene1", "gene2"])
    edge_df = pd.DataFrame(rows).sort_values(
        by=["gene1", "gene2"]
    ).reset_index(drop=True)
    return edge_df

def get_connected_components_table(G: nx.Graph) -> pd.DataFrame:
    """
    获取连通分量信息。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    rows = []
    components = list(nx.connected_components(G))
    for comp_id, comp_nodes in enumerate(components, start=1):
        for node in sorted(comp_nodes):
            rows.append({
                "gene": node,
                "component_id": comp_id,
                "component_size": len(comp_nodes)
            })
    comp_df = pd.DataFrame(rows).sort_values(
        by=["component_size", "component_id", "gene"],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    return comp_df

def summarize_graph(G: nx.Graph) -> dict:
    """
    返回图的基本统计摘要。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "average_degree": 0.0,
            "average_clustering": 0.0,
            "n_connected_components": 0,
            "largest_component_size": 0,
        }
    density = nx.density(G)
    average_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0.0
    average_clustering = nx.average_clustering(G) if n_nodes > 1 else 0.0

    components = list(nx.connected_components(G))
    n_components = len(components)
    largest_component_size = max(len(c) for c in components) if components else 0
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "average_degree": average_degree,
        "average_clustering": average_clustering,
        "n_connected_components": n_components,
        "largest_component_size": largest_component_size,
    }

def extract_largest_component(G: nx.Graph) -> nx.Graph:
    """
    提取最大连通分量子图。
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        return nx.Graph()
    largest_nodes = max(nx.connected_components(G), key=len)
    subG = G.subgraph(largest_nodes).copy()
    return subG

def build_graph_from_edge_df(
    edge_df: pd.DataFrame,
    gene1_col: str = "gene1",
    gene2_col: str = "gene2",
    weight_col: str | None = "weight",
    negative_weight_policy: str = "abs",
) -> nx.Graph:
    """
    从边表构建 networkx 无向图。

    参数
    ----
    edge_df : pd.DataFrame
        至少包含 gene1_col 和 gene2_col；
        若提供 weight_col 且该列存在，则把该列作为边权。
    gene1_col, gene2_col : str
        边两端基因列名。
    weight_col : str | None
        边权列名；若为 None 或该列不存在，则不使用边权。
    negative_weight_policy : str
        负权处理方式：
        - "abs"   : 取绝对值
        - "keep"  : 保留原值
        - "zero"  : 负值置 0
        - "drop"  : 直接丢弃负权边

    返回
    ----
    G : nx.Graph
        构建好的无向图
    """
    if not isinstance(edge_df, pd.DataFrame):
        raise TypeError("edge_df must be a pandas DataFrame.")

    required_cols = {gene1_col, gene2_col}
    missing = required_cols - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge_df missing required columns: {missing}")

    if negative_weight_policy not in {"abs", "keep", "zero", "drop"}:
        raise ValueError(
            "negative_weight_policy must be one of: 'abs', 'keep', 'zero', 'drop'"
        )
    G = nx.Graph()
    if edge_df.empty:
        return G

    df = edge_df.copy()
    df[gene1_col] = df[gene1_col].astype(str)
    df[gene2_col] = df[gene2_col].astype(str)
    use_weight = (weight_col is not None) and (weight_col in df.columns)
    for _, row in df.iterrows():
        g1 = row[gene1_col]
        g2 = row[gene2_col]

        if use_weight:
            w = row[weight_col]

            if pd.isna(w):
                continue

            w = float(w)

            if w < 0:
                if negative_weight_policy == "abs":
                    w = abs(w)
                elif negative_weight_policy == "zero":
                    w = 0.0
                elif negative_weight_policy == "drop":
                    continue
                elif negative_weight_policy == "keep":
                    pass

            G.add_edge(g1, g2, weight=w)
        else:
            G.add_edge(g1, g2)

    return G
