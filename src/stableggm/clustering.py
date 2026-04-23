import pandas as pd
import networkx as nx
import scipy.sparse as sp
try:
    import markov_clustering as mc
except ImportError:
    mc = None
def _check_graph(G: nx.Graph):
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx Graph.")
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges.")

def graph_to_sparse_matrix(
    G: nx.Graph,
    weight_attr: str = "weight",
    negative_weight_policy: str = "abs"
):
    """
    将 networkx 图转换为 scipy sparse adjacency matrix。
    参数
    ----
    G : nx.Graph
        输入图
    weight_attr : str
        边权属性名
    negative_weight_policy : str
        负权处理策略：
        - "abs"       : 取绝对值
        - "clip_zero" : 负值设为 0
        - "keep"      : 保持原样（不推荐默认）
    返回
    ----
    matrix : scipy sparse matrix
    nodes : list[str]
        节点顺序，与矩阵行列顺序一致
    """
    _check_graph(G)
    if negative_weight_policy == "zero":
        negative_weight_policy = "clip_zero"
    if negative_weight_policy not in {"abs", "clip_zero", "keep"}:
        raise ValueError(
            "negative_weight_policy must be one of: 'abs', 'clip_zero', 'zero', 'keep'"
        )
    nodes = list(G.nodes())
    # 复制图，避免修改原图
    H = G.copy()
    for u, v, data in H.edges(data=True):
        w = float(data.get(weight_attr, 1.0))
        if negative_weight_policy == "abs":
            data[weight_attr] = abs(w)
        elif negative_weight_policy == "clip_zero":
            data[weight_attr] = max(0.0, w)
        elif negative_weight_policy == "keep":
            data[weight_attr] = w
    matrix = nx.to_scipy_sparse_array(
        H,
        nodelist=nodes,
        weight=weight_attr,
        dtype=float,
        format="csr"
    )
    matrix = sp.csr_matrix(matrix)
    return matrix, nodes

def run_mcl(
    G: nx.Graph,
    inflation: float = 2.0,
    expansion: int = 2,
    loop_value: float = 1.0,
    iterations: int = 100,
    pruning_threshold: float = 0.001,
    pruning_frequency: int = 1,
    convergence_check_frequency: int = 1,
    weight_attr: str = "weight",
    negative_weight_policy: str = "abs"
):
    """
    在图上运行 Markov Clustering (MCL)。
    """
    if mc is None:
        raise ImportError(
            "markov_clustering is not installed. "
            "Please run: python -m pip install markov_clustering"
        )
    matrix, nodes = graph_to_sparse_matrix(
        G,
        weight_attr=weight_attr,
        negative_weight_policy=negative_weight_policy
    )
    result_matrix = mc.run_mcl(
        matrix,
        expansion=expansion,
        inflation=inflation,
        loop_value=loop_value,
        iterations=iterations,
        pruning_threshold=pruning_threshold,
        pruning_frequency=pruning_frequency,
        convergence_check_frequency=convergence_check_frequency
    )
    clusters = mc.get_clusters(result_matrix)
    return {
        "matrix": matrix,
        "result_matrix": result_matrix,
        "clusters": clusters,
        "nodes": nodes,
    }

def clusters_to_membership_table(
    clusters,
    nodes
) -> pd.DataFrame:
    """
    将 MCL 输出的 clusters 转为 gene-module 对应表。
    """
    rows = []
    for module_id, cluster in enumerate(clusters, start=1):
        for idx in cluster:
            if idx < 0 or idx >= len(nodes):
                raise IndexError(
                    f"Cluster index {idx} is out of range for nodes of length {len(nodes)}"
                )
            rows.append({
                "gene": nodes[idx],
                "module_id": module_id
            })
    membership_df = pd.DataFrame(rows).sort_values(
        by=["module_id", "gene"]
    ).reset_index(drop=True)
    return membership_df

def module_summary_table(
    membership_df: pd.DataFrame,
    include_gene_list: bool = False
) -> pd.DataFrame:
    """
    根据 gene-module 表生成模块摘要表。
    参数
    ----
    membership_df : pd.DataFrame
        至少包含 gene, module_id
    include_gene_list : bool
        是否附带 genes 列
    """
    if not isinstance(membership_df, pd.DataFrame):
        raise TypeError("membership_df must be a pandas DataFrame.")
    required_cols = {"gene", "module_id"}
    missing = required_cols - set(membership_df.columns)
    if missing:
        raise ValueError(f"membership_df is missing required columns: {missing}")
    if include_gene_list:
        summary_df = (
            membership_df.groupby("module_id")["gene"]
            .agg(
                module_size="count",
                genes=lambda x: ";".join(sorted(map(str, x)))
            )
            .reset_index()
            .sort_values(by=["module_size", "module_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        summary_df = (
            membership_df.groupby("module_id")["gene"]
            .count()
            .reset_index()
            .rename(columns={"gene": "module_size"})
            .sort_values(by=["module_size", "module_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
    return summary_df

def assign_modules_to_node_table(
    node_df: pd.DataFrame,
    membership_df: pd.DataFrame
) -> pd.DataFrame:
    """
    将模块分配结果合并到节点表上。
    """
    if not isinstance(node_df, pd.DataFrame):
        raise TypeError("node_df must be a pandas DataFrame.")
    if not isinstance(membership_df, pd.DataFrame):
        raise TypeError("membership_df must be a pandas DataFrame.")
    if "gene" not in node_df.columns:
        raise ValueError("node_df must contain 'gene' column.")
    if "gene" not in membership_df.columns or "module_id" not in membership_df.columns:
        raise ValueError("membership_df must contain 'gene' and 'module_id' columns.")
    merged_df = node_df.merge(
        membership_df,
        on="gene",
        how="left"
    )
    return merged_df

def run_mcl_clustering(
    G: nx.Graph,
    inflation: float = 2.0,
    expansion: int = 2,
    loop_value: float = 1.0,
    iterations: int = 100,
    pruning_threshold: float = 0.001,
    pruning_frequency: int = 1,
    convergence_check_frequency: int = 1,
    weight_attr: str = "weight",
    negative_weight_policy: str = "abs",
    include_gene_list_in_summary: bool = False
) -> dict:
    """
    一步完成 MCL 聚类，并返回常用结果。
    """
    mcl_result = run_mcl(
        G=G,
        inflation=inflation,
        expansion=expansion,
        loop_value=loop_value,
        iterations=iterations,
        pruning_threshold=pruning_threshold,
        pruning_frequency=pruning_frequency,
        convergence_check_frequency=convergence_check_frequency,
        weight_attr=weight_attr,
        negative_weight_policy=negative_weight_policy
    )
    membership_df = clusters_to_membership_table(
        clusters=mcl_result["clusters"],
        nodes=mcl_result["nodes"]
    )
    summary_df = module_summary_table(
        membership_df,
        include_gene_list=include_gene_list_in_summary
    )
    return {
        "matrix": mcl_result["matrix"],
        "clusters": mcl_result["clusters"],
        "membership_df": membership_df,
        "summary_df": summary_df,
        "nodes": mcl_result["nodes"],
        "result_matrix": mcl_result["result_matrix"],
    }
