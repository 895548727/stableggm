import inspect

import pandas as pd

import stableggm
from stableggm.pipeline import run_stableggm_pipeline
from stableggm.preprocess import preprocess_expression


def test_package_imports():
    assert stableggm.__version__ == "0.1.0"


def test_preprocess_smoke():
    expr_df = pd.DataFrame(
        {
            "sample_1": [10, 0, 5],
            "sample_2": [12, 0, 7],
            "sample_3": [11, 0, 6],
        },
        index=["gene_a", "gene_b", "gene_c"],
    )

    result = preprocess_expression(
        expr_df,
        data_type="RNA-seq",
        normalization="CPM",
        min_expression=1.0,
        zero_threshold=0.5,
    )

    assert list(result.index) == ["gene_a", "gene_c"]
    assert result.shape == (2, 3)
    assert result.isna().sum().sum() == 0


def test_pipeline_keeps_backward_compatible_parameters():
    params = inspect.signature(run_stableggm_pipeline).parameters
    assert "min_expression" in params
    assert "score_threshold" in params
