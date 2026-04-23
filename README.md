# StableGGM

StableGGM is a Python toolkit for stability-aware gene network inference from transcriptomic data. It combines preprocessing, partial-correlation estimation, subsampling-based aggregation, edge selection, network construction, module detection, plotting, and optional enrichment analysis in one package.

## Project status

This repository is still in an early stage. The package structure is now ready for GitHub publication, but the scientific workflow still needs broader debugging and validation before it should be treated as production-ready.

## Current capabilities

- RNA-seq and microarray preprocessing
- Partial-correlation estimation from expression matrices
- Stability selection through repeated subsampling
- Edge selection with Python-based FDR control
- Graph construction and network summaries
- MCL-based module detection
- Plotting helpers for diagnostics and network views
- Optional enrichment analysis with user-supplied annotations

## Installation

Basic install:

```bash
pip install -e .
```

If you need ComBat batch correction support:

```bash
pip install -e ".[batch]"
```

Development install:

```bash
pip install -e ".[dev]"
```

## Package layout

- `src/stableggm/`: publishable Python package
- `tests/`: local tests and example-driven checks
- `examples/`: demo scripts and generated artifacts
- `data/`: local data helpers and mapping scripts

## Minimal usage

```python
import pandas as pd
from stableggm.pipeline import run_stableggm_pipeline

expr_df = pd.read_csv("tests/data/gene_expression_Acinetobacter_baumannii.csv", index_col=0)

result = run_stableggm_pipeline(
    expr_df=expr_df,
    output_dir="output",
    bacteria="demo",
    data_type="RNA-seq",
    normalization="CPM",
    make_plots=False,
    annotation_df=None,
)

print(result["summary"])
```

## CLI usage

After installation, you can run the packaged command directly:

```bash
stableggm run --expr tests/data/gene_expression_Acinetobacter_baumannii.csv --output output --bacteria demo --smoke --smoke-n-genes 100 --smoke-n-samples 30 --no-make-plots
```

You can also run the module form:

```bash
python -m stableggm run --expr tests/data/gene_expression_Acinetobacter_baumannii.csv --output output --bacteria demo --smoke --smoke-n-genes 100 --smoke-n-samples 30 --no-make-plots
```

## Before uploading to GitHub

- Review and trim any large generated result folders you do not want in the first commit.
- Replace placeholder metadata if you want a real author name or repository URL.
- Run a smoke test in your target Python environment.
- Create a fresh GitHub repository, then push this folder after `git init`.

## Known limitations

- The repository still contains historical example outputs outside the package source tree.
- Several old example scripts and tests appear to target earlier APIs and may need alignment.
- Full pipeline validation has not been completed yet on this machine.

## License

MIT
