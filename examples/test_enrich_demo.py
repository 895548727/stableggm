import os
import matplotlib
matplotlib.use("Agg")

import pandas as pd

from stableggm.enrich import run_enrichment, enrich_modules, add_term_names
from stableggm.plotting import plot_enrichment_bubble
from stableggm.pipeline import _run_and_plot_enrichment
anno_df = pd.read_csv("../tests/ab_output_go_kegg_mapped.tsv", sep="\t")
membership_df = pd.read_csv("../examples/benchmark_output/Acinetobacter_baumannii_module_membership.csv")
# 简单测试数据
expr_df = pd.read_csv(
        "../tests/data/gene_expression_Acinetobacter_baumannii_2.csv",
        index_col=0
    )
output_dir="tests/test_outputs"
background_genes = list(expr_df.index)
module_enrichment_results = _run_and_plot_enrichment(
    membership_df=membership_df,
    annotation_df=anno_df,
    output_dir=output_dir,
    bacteria="bacteria",
    background_genes=background_genes,
    fdr_alpha=0.05,
);