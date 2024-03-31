
import pandas as pd
import numpy as np
import h5py
import seaborn as sns
import mofax as mfx
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['figure.dpi'] = 400

m = mfx.mofa_model("C:/Users/patri/Coding/SystemsBiology__PPCA/model/pbmc3k_nogroup_expectations.hdf5")

# ++++++++ Print some basic information ++++++

print(f"""\
Cells: {m.shape[0]}
Features: {m.shape[1]}
Groups of cells: {', '.join(m.groups)}
Views: {', '.join(m.views)}
""")

# Data structure
print("HDF5 group:\n", m.weights)
# DataFrame Structure
print("\npd.DataFrame:\n", m.get_weights(df=True).iloc[:3,:9])


# Get R-squared values for each factor
r2_df = m.get_r2(factors=list(range(8)))

# Extract R-squared values and convert them to a list
r2_values = r2_df['R2'].tolist()

# Create a list of factor names
factor_names = ["Factor_" + str(i) for i in range(8)]
# ++++++++ Plotting ++++++
# Bar plot of R-squared values for each factor
plt.figure(figsize=(10, 6))
plt.bar(factor_names, r2_values)
plt.xlabel('Factors')
plt.ylabel('R-squared values')
plt.title('R-squared values for each factor (Bar Plot)')
plt.savefig("C:/Users/patri/Coding/SystemsBiology__PPCA/figures/R2barplot.png")
plt.show()



# Heatmap of factor weights
mfx.plot_weights_heatmap(m, n_features=20,
                         factors=range(0, 9),
                         xticklabels_size=6, w_abs=True,
                         cmap="viridis", cluster_factors=False)
plt.savefig("C:/Users/patri/Coding/SystemsBiology__PPCA/figures/heatmap.png")
plt.show()


# ++++++++ All Factor weights ++++++++
nf = 2
f, axarr = plt.subplots(nf, nf, figsize=(14,14))
fnum = 0
for i in range(nf):
    for j in range(nf):
        mfx.plot_weights_scaled(m, x=fnum, y=fnum+1, n_features=10, ax=axarr[i][j])
        fnum+=2
plt.savefig("C:/Users/patri/Coding/SystemsBiology__PPCA/figures/weights.png")
plt.show()

#
'''
# Absolute dotplot of factor loadings
mfx.plot_weights_dotplot(m, n_features=3,
                         w_abs=True,
                         factors=8)
plt.show()
'''
'''
#W to Dataframe
gene_names = ["Gene_" + str(i) for i in range(W.shape[1])]
V=W.T
factor_names = ["Factor_" + str(i) for i in range(V.shape[1])]
W_df = pd.DataFrame(V, columns=factor_names, index=gene_names)

# Number of top features to select
top_n = 10

# For each factor, select the top N features by absolute loading
top_features = {}
for factor in factor_names:
    top_features[factor] = W_df[factor].abs().nlargest(top_n).index.tolist()

# Plot absolute loadings of top features for each factor
plt.figure(figsize=(10, 6))
for factor in factor_names:
    plt.plot(W_df.loc[top_features[factor], factor].abs(), label=factor)
plt.xlabel('Features')
plt.ylabel('Absolute Loadings')
plt.legend()
plt.show()
'''
# ++++++++ Reconstruction and RMSE ++++++++

# Get factor loadings (W) and factor scores (Z)
with h5py.File("C:/Users/patri/Coding/SystemsBiology__PPCA/model/pbmc3k_nogroup_expectations.hdf5", 'r') as f:
    W = f['expectations']['W']['rna'][:]
    Z = f['expectations']['Z']['group1'][:]
# Reconstruct the original data
reconstructed_data = np.dot(Z.T, W)

# Convert the reconstructed data to a DataFrame
cell_names = ["Cell_" + str(i) for i in range(reconstructed_data.shape[0])]
gene_names = ["Gene_" + str(i) for i in range(reconstructed_data.shape[1])]
df = pd.DataFrame(reconstructed_data, index=cell_names, columns=gene_names)

# Plot a heatmap of the reconstructed data
#sns.heatmap(df)
#plt.show()


