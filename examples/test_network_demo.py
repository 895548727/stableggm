import pandas as pd

from stableggm.network import (
    build_graph_from_edges,
    get_degree_table,
    get_weighted_degree_table,
    get_node_table,
    get_edge_table,
    get_connected_components_table,
    summarize_graph,
    extract_largest_component,
)

# 构造一个简单 stable_edges
stable_edges = pd.DataFrame({
    "gene1": ["A", "A", "B", "D"],
    "gene2": ["B", "C", "C", "E"],
    "weight": [0.42, -0.18, 0.35, 0.50],
    "presence_count": [9, 8, 7, 6],
    "presence_ratio": [0.9, 0.8, 0.7, 0.6]
})

print("输入边表：")
print(stable_edges)

G = build_graph_from_edges(stable_edges)

print("\n网络摘要：")
print(summarize_graph(G))

print("\nDegree table:")
print(get_degree_table(G))

print("\nWeighted degree table:")
print(get_weighted_degree_table(G))

print("\nNode table:")
print(get_node_table(G))

print("\nEdge table:")
print(get_edge_table(G))

print("\nConnected components:")
print(get_connected_components_table(G))

largest_G = extract_largest_component(G)
print("\n最大连通分量摘要：")
print(summarize_graph(largest_G))