import math
import numpy as np
import pandas as pd
from .pcor import compute_partial_correlation, pcor_to_edge_list

def _validate_subsample_input(
    expr_df: pd.DataFrame,
    subset_size: int,
    n_iterations: int
):
    if not isinstance(expr_df, pd.DataFrame):
        raise TypeError("expr_df must be a pandas DataFrame.")
    if expr_df.empty:
        raise ValueError("expr_df is empty.")
    n_genes, n_samples = expr_df.shape
    if subset_size < 2:
        raise ValueError("subset_size must be at least 2.")
    if subset_size > n_genes:
        raise ValueError("subset_size cannot be larger than number of genes.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1.")
    return n_genes, n_samples

# ...根据样本数和基因数，求迭代次数
def estimate_iterations(gene_count: int, subset_size: int) -> int:
    """
    根据基因总数和每轮抽取基因数，估计所需迭代次数。
    公式：
        upper = log10(2 / g^2)
        down  = log10(1 - s^2 / g^2)
        N     = ceil(upper / down)
    """
    gene_count = int(gene_count)
    subset_size = int(subset_size)
    if gene_count < 2:
        raise ValueError("gene_count must be >= 2.")
    if subset_size < 2:
        raise ValueError("subset_size must be >= 2.")
    if subset_size > gene_count:
        raise ValueError("subset_size cannot be larger than gene_count.")
    ratio = (subset_size ** 2) / (gene_count ** 2)
    upper = math.log10(2 / (gene_count ** 2))
    down = math.log10(1 - ratio)
    count = math.ceil(upper / down)
    # print(count)
    return count

def solve_subset_size_for_target_iterations(
    gene_count: int,
    target_iterations: int
) -> int:
    """
    反推达到目标迭代次数所需的 subset_size。
    近似反解：
        s = g * sqrt(1 - (2/g^2)^(1/N))
    """
    gene_count = int(gene_count)
    target_iterations = int(target_iterations)
    if gene_count < 2:
        raise ValueError("gene_count must be >= 2.")
    if target_iterations < 1:
        raise ValueError("target_iterations must be >= 1.")
    g = float(gene_count)
    N = float(target_iterations)

    s = g * math.sqrt(1 - (2 / (g * g)) ** (1 / N))
    return max(2, math.ceil(s))

def choose_subsample_plan(
    gene_count: int,
    sample_count: int,
    iteration_cap: int = 2500,
    iteration_trigger: int = 3000,
    max_multiplier: float = 2.0
) -> dict:
    """
    自动选择 subset_size 和 n_iterations。
    规则：
    1) 默认 subset_size = sample_count
    2) 若理论次数 <= iteration_cap，直接用理论值
    3) 若 iteration_cap < 理论次数 <= iteration_trigger，直接截到 iteration_cap
    4) 若理论次数 > iteration_trigger，使用反推法估计更大的 subset_size，
       但倍数不超过 max_multiplier，然后重新估计迭代次数，最终也不超过 iteration_cap
    """
    gene_count = int(gene_count)
    sample_count = int(sample_count)
    # print("test:",sample_count)
    if gene_count < 2:
        raise ValueError("gene_count must be >= 2.")
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2.")
    if sample_count > gene_count:
        raise ValueError("sample_count cannot be larger than gene_count.")
    default_subset_size = sample_count
    estimated_iterations = estimate_iterations(
        gene_count=gene_count,
        subset_size=default_subset_size
    )
    if estimated_iterations <= iteration_cap:
        return {
            "subset_size": default_subset_size,
            "n_iterations": estimated_iterations,
            "estimated_iterations": estimated_iterations,
            "recalculated_iterations": estimated_iterations,
            "multiplier": 1.0,
            "capped": False,
            "strategy": "default_exact"
        }
    if estimated_iterations <= iteration_trigger:
        return {
            "subset_size": default_subset_size,
            "n_iterations": iteration_cap,
            "estimated_iterations": estimated_iterations,
            "recalculated_iterations": estimated_iterations,
            "multiplier": 1.0,
            "capped": True,
            "strategy": "default_capped_to_iteration_cap"
        }
    target_subset_size = solve_subset_size_for_target_iterations(
        gene_count=gene_count,
        target_iterations=iteration_cap
    )
    raw_multiplier = target_subset_size / sample_count
    multiplier = min(raw_multiplier, max_multiplier)

    subset_size = max(2, math.ceil(multiplier * sample_count))
    subset_size = min(subset_size, gene_count)
    recalculated_iterations = estimate_iterations(
        gene_count=gene_count,
        subset_size=subset_size
    )

    n_iterations = min(recalculated_iterations, iteration_cap)
    return {
        "subset_size": subset_size,
        "n_iterations": n_iterations,
        "estimated_iterations": estimated_iterations,
        "recalculated_iterations": recalculated_iterations,
        "multiplier": multiplier,
        "capped": True,
        "strategy": "inverse_with_multiplier_cap"
    }

def random_subsample_genes(
    expr_df: pd.DataFrame,
    subset_size: int,
    random_state: int = None
) -> pd.DataFrame:
    """
    随机抽取一部分基因，返回子矩阵。
    """
    rng = np.random.default_rng(random_state)
    # print("index:", expr_df.index)
    # print("samples:",subset_size)
    selected_genes = rng.choice(expr_df.index, size=subset_size, replace=False)
    # print(len(selected_genes))
    subset_df = expr_df.loc[selected_genes]
    # subset_df.to_csv("./subset_df.csv", index=False)
    return subset_df

def _standardize_edge_pair(gene1, gene2):
    """
    统一无向边顺序，确保 (A, B) 和 (B, A) 被视为同一条边。
    """
    g1 = str(gene1)
    g2 = str(gene2)
    return (g1, g2) if g1 <= g2 else (g2, g1)

def _update_aggregate_dict(
    agg_dict: dict,
    edges_df: pd.DataFrame,
    aggregate_mode: str = "min_abs"
):
    """
    在线更新聚合字典。

    aggregate_mode:
    - "min_abs": 保留当前最小绝对值对应的 pcor（默认）
    - "mean": 保留 pcor 的累计和，最终取平均
    - "max_abs": 保留当前最大绝对值对应的 pcor
    """
    if aggregate_mode not in {"min_abs", "mean", "max_abs"}:
        raise ValueError("aggregate_mode must be one of: 'min_abs', 'mean', 'max_abs'")

    if edges_df.empty:
        return
    for _, row in edges_df.iterrows():
        g1, g2 = _standardize_edge_pair(row["gene1"], row["gene2"])
        pcor_val = float(row["pcor"])
        abs_val = abs(pcor_val)
        key = (g1, g2)
        if key not in agg_dict:
            if aggregate_mode == "mean":
                agg_dict[key] = {
                    "pcor_sum": pcor_val,
                    "n_occurrence": 1
                }
            else:
                agg_dict[key] = {
                    "pcor": pcor_val,
                    "abs_pcor": abs_val,
                    "n_occurrence": 1
                }
        else:
            agg_dict[key]["n_occurrence"] += 1

            if aggregate_mode == "mean":
                agg_dict[key]["pcor_sum"] += pcor_val

            elif aggregate_mode == "min_abs":
                if abs_val < agg_dict[key]["abs_pcor"]:
                    agg_dict[key]["pcor"] = pcor_val
                    agg_dict[key]["abs_pcor"] = abs_val

            elif aggregate_mode == "max_abs":
                if abs_val > agg_dict[key]["abs_pcor"]:
                    agg_dict[key]["pcor"] = pcor_val
                    agg_dict[key]["abs_pcor"] = abs_val

def _aggregate_dict_to_df(
    agg_dict: dict,
    aggregate_mode: str = "min_abs"
) -> pd.DataFrame:
    """
    将在线聚合字典转换为 DataFrame。

    aggregate_mode:
    - "min_abs"
    - "mean"
    - "max_abs"
    """
    if aggregate_mode not in {"min_abs", "mean", "max_abs"}:
        raise ValueError("aggregate_mode must be one of: 'min_abs', 'mean', 'max_abs'")

    if len(agg_dict) == 0:
        return pd.DataFrame(
            columns=["gene1", "gene2", "pcor", "abs_pcor", "n_occurrence"]
        )
    rows = []
    for (gene1, gene2), info in agg_dict.items():
        if aggregate_mode == "mean":
            pcor_val = info["pcor_sum"] / info["n_occurrence"]
            abs_val = abs(pcor_val)
            rows.append({
                "gene1": gene1,
                "gene2": gene2,
                "pcor": pcor_val,
                "abs_pcor": abs_val,
                "n_occurrence": info["n_occurrence"]
            })
        else:
            rows.append({
                "gene1": gene1,
                "gene2": gene2,
                "pcor": info["pcor"],
                "abs_pcor": info["abs_pcor"],
                "n_occurrence": info["n_occurrence"]
            })
    aggregated_df = pd.DataFrame(rows).sort_values(
        by=["gene1", "gene2"]
    ).reset_index(drop=True)
    return aggregated_df

def run_subsample_pcor(
    expr_df: pd.DataFrame,
    subset_size: int = None,
    n_iterations: int = None,
    random_state: int = None,
    iteration_cap: int = 2500,
    iteration_trigger: int = 3000,
    max_multiplier: float = 2.0,
    aggregate_mode: str = "min_abs",   # 新增
    # 是否保留中间结果（调试用）
    store_pcor_matrices: bool = False,
    store_edge_lists: bool = False,
    store_sampled_genes: bool = False,
):
    """
    多轮随机子采样 + 每轮计算 partial correlation。

    省内存版：
    - 默认不保存所有轮次的 pcor 矩阵和 edge list
    - 每轮 edge list 直接在线更新聚合结果
    - 最终返回 aggregated_edge_df
    参数
    ----
    expr_df : pd.DataFrame
        行=gene, 列=sample
    subset_size : int or None
        每轮抽取多少基因；若为 None，则走自动策略
    n_iterations : int or None
        迭代次数；若为 None，则走自动策略
    random_state : int or None
        全局随机种子
    iteration_cap : int
        最大迭代次数上限
    iteration_trigger : int
        超过该值时触发反推抽样策略
    max_multiplier : float
        最大抽样倍数（相对于样本数）
    store_pcor_matrices : bool
        是否保存所有轮次 pcor 矩阵（调试用）
    store_edge_lists : bool
        是否保存所有轮次 edge list（调试用）
    store_sampled_genes : bool
        是否保存所有轮次抽样基因（调试用）
    返回
    ----
    results : dict
        包含：
        - aggregated_edge_df
        - subset_size
        - n_iterations
        - auto_plan
        - pcor_matrices (可选)
        - edge_lists (可选)
        - sampled_genes (可选)
    """
    n_genes, n_samples = expr_df.shape
    auto_plan = None
    # print(n_samples)
    # 用户未指定时，走自动策略
    if subset_size is None and n_iterations is None:
        auto_plan = choose_subsample_plan(
            gene_count=n_genes,
            sample_count=n_samples,
            iteration_cap=iteration_cap,
            iteration_trigger=iteration_trigger,
            max_multiplier=max_multiplier
        )
        subset_size = auto_plan["subset_size"]
        # print(subset_size)
        n_iterations = auto_plan["n_iterations"]
    # 只给了 subset_size，没有给 n_iterations
    elif subset_size is not None and n_iterations is None:
        n_iterations = estimate_iterations(
            gene_count=n_genes,
            subset_size=subset_size
        )
    # 只给了 n_iterations，没有给 subset_size
    elif subset_size is None and n_iterations is not None:
        subset_size = n_samples
    _validate_subsample_input(expr_df, subset_size, n_iterations)
    rng = np.random.default_rng(random_state)
    # 仅调试时才保存
    pcor_matrices = [] if store_pcor_matrices else None
    edge_lists = [] if store_edge_lists else None
    sampled_genes = [] if store_sampled_genes else None
    # 在线聚合字典
    agg_dict = {}
    for i in range(n_iterations):
        iter_seed = int(rng.integers(0, 1_000_000_000))
        subset_df = random_subsample_genes(
            expr_df=expr_df,
            subset_size=subset_size,
            random_state=iter_seed
        )
        pcor_df = compute_partial_correlation(subset_df)
        # print(pcor_df)
        edges_df = pcor_to_edge_list(pcor_df)

        # 在线更新聚合结果
        _update_aggregate_dict(
            agg_dict,
            edges_df,
            aggregate_mode=aggregate_mode
        )
        # 仅调试时保存中间结果
        if store_pcor_matrices:
            pcor_matrices.append(pcor_df)

        if store_edge_lists:
            edges_df = edges_df.copy()
            edges_df["iteration"] = i + 1
            edge_lists.append(edges_df)
        if store_sampled_genes:
            sampled_genes.append(list(subset_df.index))
    aggregated_edge_df = _aggregate_dict_to_df(
        agg_dict,
        aggregate_mode=aggregate_mode
    )
    results = {
        "aggregated_edge_df": aggregated_edge_df,
        "subset_size": subset_size,
        "n_iterations": n_iterations,
        "auto_plan": auto_plan,
        "aggregate_mode": aggregate_mode,
    }
    if store_pcor_matrices:
        results["pcor_matrices"] = pcor_matrices
    if store_edge_lists:
        results["edge_lists"] = edge_lists
    if store_sampled_genes:
        results["sampled_genes"] = sampled_genes
    return results

def concat_edge_lists(edge_lists: list[pd.DataFrame]) -> pd.DataFrame:
    """
    兼容旧逻辑：把多轮 edge list 拼接成一个总表。
    仅在 store_edge_lists=True 时才有意义。
    """
    if len(edge_lists) == 0:
        return pd.DataFrame(columns=["gene1", "gene2", "pcor", "iteration"])
    return pd.concat(edge_lists, ignore_index=True)