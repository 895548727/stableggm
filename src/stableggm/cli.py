from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .pipeline import run_stableggm_pipeline


def _read_table(path: str | Path, index_col: int | None = None) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(file_path, sep=sep, index_col=index_col)


def _load_expression_matrix(path: str | Path) -> pd.DataFrame:
    expr_df = _read_table(path, index_col=0)
    if expr_df.empty:
        raise ValueError("Expression matrix is empty.")
    return expr_df


def _load_batch_series(path: str | Path) -> pd.Series:
    batch_df = _read_table(path)
    required_cols = {"sample", "batch"}
    missing = required_cols - set(batch_df.columns)
    if missing:
        raise ValueError(f"Batch file is missing required columns: {sorted(missing)}")
    return pd.Series(
        batch_df["batch"].values,
        index=batch_df["sample"].astype(str).values,
    )


def _load_annotation_df(path: str | Path) -> pd.DataFrame:
    annotation_df = _read_table(path)
    if annotation_df.empty:
        raise ValueError("Annotation file is empty.")
    return annotation_df


def _subset_expression(
    expr_df: pd.DataFrame,
    smoke: bool,
    smoke_n_genes: int,
    smoke_n_samples: int,
) -> pd.DataFrame:
    if not smoke:
        return expr_df

    n_genes = min(int(smoke_n_genes), expr_df.shape[0])
    n_samples = min(int(smoke_n_samples), expr_df.shape[1])
    return expr_df.iloc[:n_genes, :n_samples].copy()


def _align_batch_series(batch_series: pd.Series | None, expr_df: pd.DataFrame) -> pd.Series | None:
    if batch_series is None:
        return None

    sample_names = list(map(str, expr_df.columns))
    missing = [sample for sample in sample_names if sample not in batch_series.index]
    if missing:
        raise ValueError(
            f"Batch series is missing {len(missing)} samples from expression matrix, "
            f"for example: {missing[:5]}"
        )
    return batch_series.loc[sample_names]


def _normalize_method(method: str) -> str:
    aliases = {
        "python_genenet_like": "python_genenet_like",
        "python_bh": "python_genenet_like",
        "python": "python_genenet_like",
    }
    normalized = aliases.get(method)
    if normalized is None:
        raise ValueError(
            "Unsupported method. Use one of: python_genenet_like, python_bh, python."
        )
    return normalized


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stableggm",
        description="StableGGM command line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the StableGGM pipeline on an expression matrix.",
    )
    run_parser.add_argument("--expr", required=True, help="Expression matrix CSV/TSV path.")
    run_parser.add_argument("--output", required=True, help="Output directory.")
    run_parser.add_argument(
        "--bacteria",
        default=None,
        help="Dataset label used in output file names. Defaults to the expression file stem.",
    )
    run_parser.add_argument(
        "--data-type",
        default="RNA-seq",
        choices=["RNA-seq", "microarray"],
        help="Input data type.",
    )
    run_parser.add_argument(
        "--normalization",
        default="CPM",
        help="Normalization mode. Use 'none' to skip.",
    )
    run_parser.add_argument("--batch", default=None, help="Batch CSV/TSV with sample,batch columns.")
    run_parser.add_argument(
        "--annotation",
        default=None,
        help="Optional annotation CSV/TSV passed through to enrichment.",
    )
    run_parser.add_argument(
        "--min-expression",
        type=float,
        default=None,
        help="Filter genes below the mean expression threshold before zero filtering.",
    )
    run_parser.add_argument("--zero-threshold", type=float, default=0.5)
    run_parser.add_argument(
        "--microarray-logged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether microarray inputs are already log-transformed.",
    )
    run_parser.add_argument("--n-channels", type=_positive_int, default=3)
    run_parser.add_argument("--subset-size", type=_positive_int, default=None)
    run_parser.add_argument("--n-iterations", type=_positive_int, default=None)
    run_parser.add_argument("--random-state", type=int, default=123)
    run_parser.add_argument("--iteration-cap", type=_positive_int, default=2500)
    run_parser.add_argument("--iteration-trigger", type=_positive_int, default=3000)
    run_parser.add_argument("--max-multiplier", type=float, default=2.0)
    run_parser.add_argument(
        "--intersection-mode",
        choices=["soft", "strict"],
        default="soft",
    )
    run_parser.add_argument("--min-presence", type=_positive_int, default=2)
    run_parser.add_argument(
        "--method",
        default="python_genenet_like",
        help="Edge selection method.",
    )
    run_parser.add_argument("--fdr-alpha", type=float, default=0.1)
    run_parser.add_argument("--score-threshold", type=float, default=0.9)
    run_parser.add_argument("--cutoff-ggm", type=float, default=0.9)
    run_parser.add_argument("--inflation", type=float, default=2.0)
    run_parser.add_argument(
        "--negative-weight-policy",
        choices=["abs", "keep", "zero"],
        default="abs",
    )
    run_parser.add_argument(
        "--save-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument(
        "--store-pcor-matrices",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--store-edge-lists",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--store-sampled-genes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--run-inflation-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--plot-inflation-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_parser.add_argument(
        "--plot-venn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run on the first N genes/samples only.",
    )
    run_parser.add_argument("--smoke-n-genes", type=_positive_int, default=100)
    run_parser.add_argument("--smoke-n-samples", type=_positive_int, default=30)
    run_parser.set_defaults(handler=_run_command)

    return parser

def _run_command(args: argparse.Namespace) -> int:
    expr_path = Path(args.expr)
    output_dir = Path(args.output)
    bacteria = args.bacteria or expr_path.stem

    expr_df = _load_expression_matrix(expr_path)
    expr_input = _subset_expression(
        expr_df=expr_df,
        smoke=args.smoke,
        smoke_n_genes=args.smoke_n_genes,
        smoke_n_samples=args.smoke_n_samples,
    )
    batch_series = None
    if args.batch:
        batch_series = _load_batch_series(args.batch)
        batch_series = _align_batch_series(batch_series, expr_input)

    annotation_df = _load_annotation_df(args.annotation) if args.annotation else None
    normalization = None if str(args.normalization).lower() == "none" else args.normalization
    method = _normalize_method(args.method)
    result = run_stableggm_pipeline(
        expr_df=expr_input,
        output_dir=str(output_dir),
        bacteria=bacteria,
        data_type=args.data_type,
        normalization=normalization,
        min_expression=args.min_expression,
        zero_threshold=args.zero_threshold,
        batch_series=batch_series,
        microarray_logged=args.microarray_logged,
        n_channels=args.n_channels,
        subset_size=args.subset_size,
        n_iterations=args.n_iterations,
        random_state=args.random_state,
        iteration_cap=args.iteration_cap,
        iteration_trigger=args.iteration_trigger,
        max_multiplier=args.max_multiplier,
        intersection_mode=args.intersection_mode,
        min_presence=args.min_presence,
        plot_venn=args.plot_venn,
        plot_inflation_sensitivity=args.plot_inflation_sensitivity,
        method=method,
        fdr_alpha=args.fdr_alpha,
        score_threshold=args.score_threshold,
        cutoff_ggm=args.cutoff_ggm,
        inflation=args.inflation,
        run_inflation_scan=args.run_inflation_scan,
        negative_weight_policy=args.negative_weight_policy,
        annotation_df=annotation_df,
        save_intermediate=args.save_intermediate,
        make_plots=args.make_plots,
        store_pcor_matrices=args.store_pcor_matrices,
        store_edge_lists=args.store_edge_lists,
        store_sampled_genes=args.store_sampled_genes,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
