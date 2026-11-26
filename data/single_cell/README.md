# Single-cell data and reference embeddings

| Dataset | Cells | Genes | Annotation groups | Rank | Positive scales |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSE75748time | 758 | 19,189 | 6 | 28 | 757 |
| Dendritic_batch1 | 384 | 15,426 | 4 | 20 | 383 |

- **GSE75748time** contains the time-course subset of the human embryonic stem cell differentiation study by Chu et al. (2016). The annotations identify the sampling time of each cell. [GEO GSE75748](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748) · [Study](https://doi.org/10.1186/s13059-016-1033-x)
- **Dendritic_batch1** contains the first dendritic-cell batch in the benchmark, from the human blood immune-cell study by Villani et al. (2017). The annotations identify cell populations. [GEO GSE94820](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94820) · [Study](https://doi.org/10.1126/science.aah4573)

Each `expression.csv.gz` contains the supplied processed expression matrix, with the cell identifiers in the same order as `labels.csv`. For GSE75748time, cells occupy columns. For Dendritic_batch1, cells occupy rows. The loader places cells in columns, applies `log10(1 + x)`, and divides each cell column by its Euclidean norm before fitting.

Each `config.json` records the preprocessing and fit settings. Each `reference_embeddings.npz` contains one rank-by-cell evaluation matrix for each of `NMF`, `GNMF`, `TNMF`, `kTNMF` and `pNMF`. For pNMF, the matrix is the first positive-scale embedding from the joint fit, with each row multiplied by the corresponding basis-column norm. The four baseline embeddings already incorporate this scaling.

Running `python -m examples.single_cell score` from the repository root clusters the columns of these saved embeddings. Labels supply the cluster count and the reference assignments for scoring. [Reference scores for the two datasets](reference_scores.csv)
