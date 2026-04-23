__version__ = "0.1.0"

from .preprocess import preprocess_expression

from .pcor import (
    compute_precision_matrix,
    compute_partial_correlation,
    pcor_to_edge_list,
)

from .subsample import (
    estimate_iterations,
    solve_subset_size_for_target_iterations,
    choose_subsample_plan,
    random_subsample_genes,
    run_subsample_pcor,
    concat_edge_lists,
)

from .normalization import (
    fisher_z_transform,
    normalize_pcor_column,
)

from .edge_selection import (
    matrix_to_edge_table,
    cal_net_python_genenet_like,
    select_edges,
)

from .channel_pipeline import (
    screen_df,
    run_single_channel_pipeline,
)

from .stability import (
    run_single_stability_channel,
    run_stability_selection,
    extract_stable_edge_keys,
)

from .network import (
    build_graph_from_edges,
    get_degree_table,
    get_weighted_degree_table,
    get_abs_weighted_degree_table,
    get_node_table,
    get_edge_table,
    get_connected_components_table,
    summarize_graph,
    extract_largest_component,
)

from .clustering import (
    graph_to_sparse_matrix,
    run_mcl,
    clusters_to_membership_table,
    module_summary_table,
    assign_modules_to_node_table,
    run_mcl_clustering,
)

from .enrich import (
    run_enrichment,
    enrich_modules,
    add_term_names,
)

from .plotting import (
    plot_normalization_distributions,
    plot_batch_correction_boxplots,
    draw_graph,
    draw_largest_component,
    draw_graph_by_degree,
    plot_degree_distribution,
    draw_graph_by_module,
    draw_largest_component_by_module,
    draw_module_subgraph,
    plot_module_size_distribution,
    plot_edge_weight_distribution,
    plot_edge_weight_density,
    plot_component_size_distribution,
    plot_top_degree_genes,
    plot_top_weighted_degree_genes,
    plot_presence_distribution_combined,
    plot_top_stable_edges,
    plot_enrichment_bar,
    plot_enrichment_dot,
    plot_enrichment_bubble,
    plot_mcl_inflation_sensitivity,
    plot_degree_distribution_loglog
)

from .pipeline import run_stableggm_pipeline

__all__ = [
    "__version__",
    "preprocess_expression",
    "compute_precision_matrix",
    "compute_partial_correlation",
    "pcor_to_edge_list",

    "estimate_iterations",
    "solve_subset_size_for_target_iterations",
    "choose_subsample_plan",
    "random_subsample_genes",
    "run_subsample_pcor",
    "concat_edge_lists",

    "fisher_z_transform",
    "normalize_pcor_column",

    "matrix_to_edge_table",
    "cal_net_python_genenet_like",
    "select_edges",
    "screen_df",
    "run_single_channel_pipeline",

    "run_single_stability_channel",
    "run_stability_selection",
    "extract_stable_edge_keys",

    "build_graph_from_edges",
    "get_degree_table",
    "get_weighted_degree_table",
    "get_abs_weighted_degree_table",
    "get_node_table",
    "get_edge_table",
    "get_connected_components_table",
    "summarize_graph",
    "extract_largest_component",

    "graph_to_sparse_matrix",
    "run_mcl",
    "clusters_to_membership_table",
    "module_summary_table",
    "assign_modules_to_node_table",
    "run_mcl_clustering",

    "run_enrichment",
    "enrich_modules",
    "add_term_names",

    "plot_normalization_distributions",
    "plot_batch_correction_boxplots",
    "draw_graph",
    "draw_largest_component",
    "draw_graph_by_degree",
    "plot_degree_distribution",
    "draw_graph_by_module",
    "draw_largest_component_by_module",
    "draw_module_subgraph",
    "plot_module_size_distribution",
    "plot_edge_weight_distribution",
    "plot_edge_weight_density",
    "plot_component_size_distribution",
    "plot_top_degree_genes",
    "plot_top_weighted_degree_genes",
    "plot_presence_distribution_combined",
    "plot_top_stable_edges",
    "plot_enrichment_bar",
    "plot_enrichment_dot",
    "plot_enrichment_bubble",
    "plot_mcl_inflation_sensitivity",
    "plot_degree_distribution_loglog",

    "run_stableggm_pipeline",
]
