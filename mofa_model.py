import os

import pandas as pd
import numpy as np
import scanpy as sc
from mofapy2.run.entry_point import mofa
import h5py
import seaborn


sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=300, facecolor="white")

results_file = "C:/Users/patri/Coding/SystemsBiology__PPCA/pbmc3k.h5ad"  # the file that will store the analysis results

adata = sc.read_10x_mtx(
    "data/filtered_gene_bc_matrices/hg19/",  # the directory with the `.mtx` file
    var_names="gene_symbols",  # use gene symbols for the variable names (variables-axis index)
    cache=True,  # write a cache file for faster subsequent reading
)
# Assign the current state of adata to adata.raw


sc.pl.highest_expr_genes(adata, n_top=20)
sc.pp.filter_cells(adata, min_genes=900)
sc.pp.filter_genes(adata, min_cells=42)
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
os.chdir("C:/Users/patri/Coding/SystemsBiology__PPCA/figures/")
sc.pl.highly_variable_genes(adata, save="hvg.png")

#Actual Filtering
#Slicing HVG
adata = adata[:, adata.var['highly_variable']].copy()
#Regressing mitochondrial genes
sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
#Scaling to toal counts
sc.pp.scale(adata, max_value=10)

print(adata)


#Actual MOFA
m = mofa(adata,
         expectations=["W","Z","AlphaW","AlphaZ"],
         use_raw=True,
         n_factors=2,
         outfile="C:/Users/patri/Coding/SystemsBiology__PPCA/model/pbmc3k_2Factors.hdf5", quiet=False)

f = h5py.File("C:/Users/patri/Coding/SystemsBiology__PPCA/model/pbmc3k_2Factors.hdf5")
print(f['intercepts/rna/group1'])
f.close()

file_path = "C:/Users/patri/Coding/SystemsBiology__PPCA/model/pbmc3k_2Factors.hdf5"
if os.path.isfile(file_path):
    print(f"The file exists at {file_path}")
else:
    print(f"The file does not exist at {file_path}")