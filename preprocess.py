import scanpy as sc
import pandas as pd
import pathlib

# set save path relative to this file
SAVE_PATH = pathlib.Path(__file__).parent / 'testwrite'

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

results_file = "write/pbmc3k.h5ad"  # the file that will store the analysis results

adata = sc.read_10x_mtx(
    "data/filtered_gene_bc_matrices/hg19/",  # the directory with the `.mtx` file
    var_names="gene_symbols",  # use gene symbols for the variable names (variables-axis index)
    cache=True,  # write a cache file for faster subsequent reading
)

adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`

# adata.write_csvs(dirname=SAVE_PATH, skip_data=False)

#df = pd.DataFrame(adata.X)
#print(df.head())
#print(df.shape)
#df.to_csv(SAVE_PATH / 'pbmc3k.csv', index=False, sep=',')
print(adata)

#sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", color="CST3")
#sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")
#adata = adata[adata.obs.n_genes_by_counts < 2500, :]
#adata = adata[adata.obs.pct_counts_mt < 5, :].copy()


sc.pl.highest_expr_genes(adata, n_top=20)
sc.pp.filter_cells(adata, min_genes=900)
sc.pp.filter_genes(adata, min_cells=42)
print(adata.obs.keys())
print(adata.var.keys())
#annotate mitochandrial genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)
sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()


print(adata)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata.raw = adata

sc.pp.highly_variable_genes(adata, min_mean=0.0250, max_mean=3, min_disp=0.6)
#adata.layers["counts"] = adata.X.copy()
'''sc.pp.highly_variable_genes(
    adata,
    n_top_genes=1200,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
)'''
sc.pl.highly_variable_genes(adata)
print(adata)


#adata.n_top_genes.write_csvs('/testwrite', skip_data=True, sep=',')

# Filter the adata object to include only highly variable genes
highly_variable_genes = adata[:, adata.var['highly_variable']]
print(highly_variable_genes)
# Convert the filtered data to a dense matrix
dense_matrix = highly_variable_genes.X.todense()

# Convert the dense matrix to a DataFrame
df = pd.DataFrame(dense_matrix, columns=highly_variable_genes.var_names)
# Export the DataFrame to a CSV file
df.to_csv('C:/Users/patri/Coding/SystemsBiology__PPCA/data/highly_variable_genes.csv', index=False)


# +++++ PCA +++++

sc.tl.pca(adata, svd_solver="arpack")

#Plotting PCA + Variance ratio
sc.pl.pca(adata, color="CST3")
sc.pl.pca_variance_ratio(adata, log=True)