# StableGGM

**StableGGM** is a Python package for stability-oriented gene network inference and analysis from transcriptomic data.  
It provides an end-to-end workflow from expression matrix input to stable edge selection, network construction, module detection, enrichment analysis, and publication-ready visualizations.

## Features

- Preprocessing for RNA-seq and microarray expression matrices
- Partial-correlation based network inference
- Stability selection through repeated subsampling and multi-channel aggregation
- Python-based edge selection with FDR control
- Graph construction and network summary statistics
- MCL-based module detection
- Diagnostic and network visualization utilities
- Optional GO/KEGG enrichment analysis with user-supplied annotations
- Command-line interface for running the full pipeline

---
## Installation

### Basic installation

```bash
pip install stableggm
```

### Install from source

```bash
git clone https://github.com/895548727/stableggm.git
cd stableggm
pip install -e .
```
🚨 IMPORTANT: Demonstration Datasets

To run the quick start and minimal examples below, please download the required empirical Acinetobacter baumannii datasets from our repository:

    📊 Expression Matrix: gene_expression_Acinetobacter_baumannii.csv(https://github.com/895548727/stableggm/blob/main/gene_expression_Acinetobacter_baumannii.csv) — Raw/processed transcriptomic expression profiles.

    🏷️ Batch Information: ab_batch_expanded_clean.csv(https://github.com/895548727/stableggm/blob/main/ab_batch_expanded_clean.csv) — Sample batch metadata for batch-correction utilities.

    🧬 Functional Annotation: ab_output_go_kegg_mapped.tsv(https://github.com/895548727/stableggm/blob/main/ab_output_go_kegg_mapped.tsv) — User-supplied GO/KEGG functional mappings for downstream enrichment analysis.

Please ensure these files are placed in your current working directory before executing the pipeline.

## Quick start (Python API)

```python
import pandas as pd
from stableggm.pipeline import run_stableggm_pipeline

expr_df = pd.read_csv("gene_expression_Acinetobacter_baumannii.csv", index_col=0)

result = run_stableggm_pipeline(
    expr_df=expr_df,
    output_dir="stableggm_output",
    bacteria="Acinetobacter_baumannii",
    data_type="RNA-seq",
    normalization="CPM",
    make_plots=True,
    annotation_df=None,
)

print(result["summary"])
```

---

## Command-line usage

After installation, the package provides a command-line interface:

```bash
stableggm --help
stableggm run --help
```

### Minimal example

```bash
stableggm run \
  --expr gene_expression_Acinetobacter_baumannii.csv \
  --output stableggm_output \
  --data-type RNA-seq \
  --normalization CPM
```

### Example with smoke test

```bash
stableggm run \
  --expr gene_expression_Acinetobacter_baumannii.csv \
  --output stableggm_output \
  --data-type RNA-seq \
  --normalization CPM \
  --smoke \
  --smoke-n-genes 100 \
  --smoke-n-samples 30 \
  --no-make-plots
```

### Example with microarray data, batch correction, and annotation

```bash
stableggm run \
  --expr gene_expression_Acinetobacter_baumannii.csv \
  --output stableggm_output \
  --data-type microarray \
  --no-microarray-logged \
  --batch ab_batch_expanded_clean.csv \
  --annotation ab_output_go_kegg_mapped.tsv
```

---

## Main outputs

Running the pipeline can generate:

- preprocessed expression matrix
- stability table
- stable edge table
- graph summary table
- node and edge tables
- module membership table
- module summary table
- optional enrichment results
- publication-ready figures

Typical plots include:

- batch-correction boxplots
- batch-correction PCA plots
- edge-overlap Venn plots
- degree distribution plots
- edge-weight distribution plots
- largest connected component plots
- module-colored network plots
- module subgraph plots
- hub-gene plots
- enrichment bubble plots

---

## Input requirements

### Expression matrix

- rows: genes
- columns: samples
- file format: CSV or TSV

### Batch file (optional)

Must contain at least two columns:

- `sample`
- `batch`

### Annotation file (optional)

Used for enrichment analysis.  
The supported annotation format should match the pipeline requirements in your installed version.  
Typical fields include gene identifiers and GO/KEGG annotations.

---

## Package structure

```text
stableggm/
├── src/stableggm/
├── tests/
├── examples/
├── README.md
├── LICENSE
└── pyproject.toml
```

---

## Example workflow

StableGGM supports the following workflow:

1. expression preprocessing  
2. repeated subsampling and partial-correlation inference  
3. online edge aggregation  
4. stable edge selection across channels  
5. graph construction  
6. module detection  
7. optional enrichment analysis  
8. automated plotting and result export  

---

## Development installation

For local development:

```bash
pip install -e ".[dev]"
```

---

## License

MIT License
