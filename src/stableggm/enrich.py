from __future__ import annotations

import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

def _validate_annotation_df(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """
    检查注释表格式。
    需要至少两列：
    - gene
    - term
    """
    if not isinstance(annotation_df, pd.DataFrame):
        raise TypeError("annotation_df must be a pandas DataFrame.")
    required_cols = {"gene", "term"}
    missing = required_cols - set(annotation_df.columns)
    if missing:
        raise ValueError(f"annotation_df is missing required columns: {missing}")
    df = annotation_df.copy()
    df["gene"] = df["gene"].astype(str)
    df["term"] = df["term"].astype(str)
    return df

def run_enrichment(
    gene_list: list[str],
    annotation_df: pd.DataFrame,
    background_genes: list[str] | None = None,
    min_term_size: int = 3,
    max_term_size: int | None = None,
    fdr_alpha: float = 0.05
) -> pd.DataFrame:
    """
    通用富集分析函数（GO / KEGG 都可用）。
    参数
    ----
    gene_list : list[str]
        待分析基因集
    annotation_df : pd.DataFrame
        注释表，至少两列：
        - gene
        - term
        可额外有 term_name, source 等列
    background_genes : list[str] or None
        背景基因集；若为 None，则用 annotation_df 中所有基因
    min_term_size : int
        最小 term 大小
    max_term_size : int or None
        最大 term 大小
    fdr_alpha : float
        FDR 阈值
    返回
    ----
    enrich_df : pd.DataFrame
        富集结果表
    """
    annotation_df = _validate_annotation_df(annotation_df)
    gene_list = [str(g) for g in gene_list]
    gene_set = set(gene_list)
    if background_genes is None:
        background_set = set(annotation_df["gene"])
    else:
        background_set = set(str(g) for g in background_genes)
    # 限制在背景内
    gene_set = gene_set.intersection(background_set)
    if len(gene_set) == 0:
        return pd.DataFrame(
            columns=[
                "term", "term_size", "overlap_size", "query_size",
                "background_size", "pvalue", "qvalue", "score", "overlap_genes"
            ]
        )
    annotation_df = annotation_df.loc[
        annotation_df["gene"].isin(background_set)
    ].copy()
    term_groups = annotation_df.groupby("term")["gene"].apply(set)
    M = len(background_set)   # 背景总基因数
    N = len(gene_set)         # query gene 数
    rows = []
    for term, term_genes in term_groups.items():
        term_genes = set(term_genes)
        n = len(term_genes)   # 该 term 在背景中的基因数
        if n < min_term_size:
            continue
        if max_term_size is not None and n > max_term_size:
            continue
        overlap_genes = gene_set.intersection(term_genes)
        k = len(overlap_genes)
        if k == 0:
            continue
        # P(X >= k)
        pval = hypergeom.sf(k - 1, M, n, N)
        rows.append({
            "term": term,
            "term_size": n,
            "overlap_size": k,
            "gene_count": k,
            "query_size": N,
            "background_size": M,
            "pvalue": pval,
            "overlap_genes": ";".join(sorted(overlap_genes))
        })
    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "term", "term_size", "overlap_size", "query_size",
                "background_size", "pvalue", "qvalue", "score", "overlap_genes"
            ]
        )
    enrich_df = pd.DataFrame(rows)
    _, qvals, _, _ = multipletests(
        enrich_df["pvalue"].values,
        alpha=fdr_alpha,
        method="fdr_bh"
    )
    enrich_df["qvalue"] = qvals
    enrich_df["score"] = -enrich_df["qvalue"].clip(lower=1e-300).apply(__import__("math").log10)
    enrich_df["gene_count"] = enrich_df["overlap_size"]
    enrich_df["rich_factor"] = enrich_df["overlap_size"] / enrich_df["term_size"]
    enrich_df["gene_ratio"] = enrich_df["overlap_size"] / enrich_df["query_size"]
    enrich_df["neg_log10_qvalue"] = -enrich_df["qvalue"].clip(lower=1e-300).apply(__import__("math").log10)
    enrich_df = enrich_df.sort_values(
        by=["qvalue", "pvalue", "overlap_size"],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    return enrich_df

def enrich_modules(
    membership_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    background_genes: list[str] | None = None,
    min_module_size: int = 3,
    min_term_size: int = 3,
    max_term_size: int | None = None,
    fdr_alpha: float = 0.05,
    top_n_modules: int = 20   # 新增参数：只分析最大的前 N 个模块
) -> dict[int, pd.DataFrame]:
    """
    对每个 module 分别做富集分析（可选择只分析最大的前 N 个模块）。
    参数
    ----
    membership_df : pd.DataFrame
        至少包含：gene, module_id
    annotation_df : pd.DataFrame
        至少包含：gene, term
    background_genes : list[str] or None
        背景基因集
    min_module_size : int
        模块最小基因数（小于该值的模块将被忽略）
    min_term_size : int
        每个 term 在背景中的最小基因数
    max_term_size : int or None
        每个 term 在背景中的最大基因数
    fdr_alpha : float
        FDR 阈值
    top_n_modules : int
        只对基因数最多的前 top_n_modules 个模块进行富集分析。
        若为 0 或 None，则分析所有满足 min_module_size 的模块。
        若指定数值大于实际模块数，则分析所有模块。
    返回
    ----
    result_dict : dict[int, pd.DataFrame]
        模块ID -> 富集结果表
    """
    # 参数校验
    if not isinstance(membership_df, pd.DataFrame):
        raise TypeError("membership_df must be a pandas DataFrame.")
    required_cols = {"gene", "module_id"}
    missing = required_cols - set(membership_df.columns)
    if missing:
        raise ValueError(f"membership_df is missing required columns: {missing}")
    # 1. 计算每个模块的基因数
    module_sizes = membership_df.groupby("module_id")["gene"].count()
    # 过滤掉小于 min_module_size 的模块
    valid_modules = module_sizes[module_sizes >= min_module_size].index
    if len(valid_modules) == 0:
        return {}
    # 2. 按基因数降序排序，选出前 top_n_modules 个模块
    sorted_modules = module_sizes.loc[valid_modules].sort_values(ascending=False)
    if top_n_modules and top_n_modules > 0:
        selected_modules = sorted_modules.head(top_n_modules).index.tolist()
    else:
        selected_modules = sorted_modules.index.tolist()
    # 3. 仅保留选中的模块的基因数据
    membership_filtered = membership_df[membership_df["module_id"].isin(selected_modules)]
    # 4. 分组并对每个选中的模块进行富集分析
    result_dict = {}
    module_groups = membership_filtered.groupby("module_id")["gene"].apply(list)
    for module_id, genes in module_groups.items():
        enrich_df = run_enrichment(
            gene_list=genes,
            annotation_df=annotation_df,
            background_genes=background_genes,
            min_term_size=min_term_size,
            max_term_size=max_term_size,
            fdr_alpha=fdr_alpha
        )
        result_dict[module_id] = enrich_df
    return result_dict

def add_term_names(
    enrich_df: pd.DataFrame,
    term_info_df: pd.DataFrame,
    term_col: str = "term",
    name_col: str = "term_name"
) -> pd.DataFrame:
    """
    将 term 名称信息合并到富集结果中。
    term_info_df 至少应包含：
    - term
    - term_name
    """
    if not isinstance(enrich_df, pd.DataFrame):
        raise TypeError("enrich_df must be a pandas DataFrame.")
    if not isinstance(term_info_df, pd.DataFrame):
        raise TypeError("term_info_df must be a pandas DataFrame.")
    if term_col not in enrich_df.columns:
        raise ValueError(f"'{term_col}' not found in enrich_df.")
    if term_col not in term_info_df.columns or name_col not in term_info_df.columns:
        raise ValueError(f"term_info_df must contain '{term_col}' and '{name_col}'.")
    merged = enrich_df.merge(
        term_info_df[[term_col, name_col]],
        on=term_col,
        how="left"
    )
    return merged
