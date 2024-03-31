from pca_functions import pca
from pca_functions import ppca
import numpy as np
from mofapy2.run.entry_point import mofa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import math
from joblib import dump, load
import joblib
import anndata
import seaborn as sns

def RMSE(original_data,recon_data):
    # Calculate the squared differences between corresponding data points
    squared_diff = [(np.array(og) - np.array(rec)) ** 2
                    for og, rec in zip(original_data, recon_data)]
    # Calculate the mean of the squared differences
    mse = (np.mean(squared_diff))**(1/2)

    return mse
#original data
data = pd.read_csv('C:/Users/patri/Coding/SystemsBiology__PPCA/data/highly_variable_genes.csv')
n_samples,n_features=data.shape

#W_init = np.random.randn(data.shape[1], 2)
#pca(data, 2)
#ppca(data, 2, W_init, 1)

def mofa_r(data, Nfactors):
    adata =  anndata.AnnData(data)
    m = mofa(adata,
         expectations=["W","Z","AlphaW","AlphaZ"],
         n_factors=Nfactors,
        outfile="C:/Users/patri/Coding/SystemsBiology__PPCA/model/run_miss.hdf5")
    with h5py.File("C:/Users/patri/Coding/SystemsBiology__PPCA/model/run_miss.hdf5", 'r') as f:
        W = f['expectations']['W']['rna'][:]
        Z = f['expectations']['Z']['group1'][:]
    recon_data_mofa = np.asarray(np.dot(Z.T, W) + np.nanmean(np.asarray(data), axis=0))
    return m, recon_data_mofa


# Get factor loadings (W) and factor scores (Z)
#with h5py.File("C:/Users/patri/Coding/SystemsBiology__PPCA/model/pbmc3k_2Factors.hdf5", 'r') as f:
#    W = f['expectations']['W']['rna'][:]
#    Z = f['expectations']['Z']['group1'][:]
# Reconstruct the original data
#reconstructed_data = np.asarray(np.dot(Z.T, W) + np.nanmean(np.asarray(data), axis=0))

# Convert the reconstructed data to a DataFrame
#cell_names = ["Cell_" + str(i) for i in range(reconstructed_data.shape[0])]
#gene_names = ["Gene_" + str(i) for i in range(reconstructed_data.shape[1])]
#recon_data_mofa = pd.DataFrame(reconstructed_data, index=cell_names, columns=gene_names)

# Plot a heatmap of the reconstructed data
#sns.heatmap(df)
#plt.show()

# Calculate the root mean squared error (RMSE) between the original and reconstructed data


def replace_with_nans(data, fraction):
    result = data.copy()  # Make a copy to avoid modifying the original array
    n_samples,n_features = data.shape
    for i in range(n_samples):
        # Choose one or two random indices in each row
        idx = np.random.choice(n_features, size=math.ceil(n_features*fraction/100), replace=False)
        # Replace the values at those indices with NaN
        result[i, idx] = np.nan
    return result

def test_for_mising_data(data, n_comp, ax, anno=False, load=False):
    # fraction=[0.1,0.3,0.6,1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,99]
    fraction = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    if load:
        y1 = joblib.load('y1_vary_miss.joblib')
        y2 = joblib.load('y2_vary_miss.joblib')
        y3 = joblib.load('y3_vary_miss.joblib')
    else:
        y1 = []
        y2 = []
        y3 = []
        for f in fraction:
            compromised_data = replace_with_nans(np.asarray(data), f)
            pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca = pca(compromised_data, n_comp)
            W_init = np.random.randn(data.shape[1], n_comp)
            s_init = 1
            ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(compromised_data, n_comp, W_init, s_init)
            m, recon_data_mofa = mofa_r(compromised_data, n_comp)
            y1.append(RMSE(np.asarray(data), recon_data_pca))
            y2.append(RMSE(np.asarray(data), recon_data_ppca))
            y3.append(RMSE(np.asarray(data), recon_data_mofa))
        joblib.dump(y1,'y1_vary_miss.joblib')
        joblib.dump(y2,'y2_vary_miss.joblib')
        joblib.dump(y3,'y3_vary_miss.joblib')


    ax.plot(fraction, y1, label='PCA')
    ax.plot(fraction, y2, label='PPCA')
    ax.plot(fraction, y3, label='MOFA')
    ax.set_xlabel('fraction of observables missing per row')
    ax.set_ylabel('Rooted mean squared error')
    ax.legend(loc='upper center')
    if anno != False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')


def test_for_mising_data_vary_comp(data,ax,anno=False,load=False):
    #fraction=[0.1,0.3,0.6,1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,99]
    num_comp=np.arange(2,200,20)
    fraction=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    if load:
        yy2=joblib.load('yy2_vary_miss.joblib')
    else:
        yy2=[]
        for n in num_comp:
            y2=[]
            for f in fraction:
                compromised_data=replace_with_nans(np.asarray(data),f)
                #pca_data, pca_eig_vecs, pca_eig_vals, recon_data_pca=pca(compromised_data,n)
                W_init = np.random.randn(data.shape[1], n)
                s_init=1
                ppca_data, W, s_squared, E_xt, recon_data_ppca, iter = ppca(compromised_data, n,W_init,s_init)
                m, recon_data_mofa = mofa_r(compromised_data, n)
                #RMSE_pca=RMSE(np.asarray(data),recon_data_pca)
                RMSE_ppca=RMSE(np.asarray(data),recon_data_ppca)
                RMSE_mofa = RMSE(np.asarray(data), recon_data_mofa)
                y2.append(RMSE_ppca-RMSE_mofa)
            yy2.append(y2)
        joblib.dump(yy2,'yy2_vary_miss.joblib')
    for i,y in enumerate(yy2):
        ax.plot(fraction, y,label='n.c.:'+str(num_comp[i]))

    ax.set_xlabel('fraction of observables missing per row')
    ax.set_ylabel('Rooted mean squared error')
    ax.legend(loc='lower center',ncol=4)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    if anno!= False:
        ax.annotate(anno, xy=(0.02, 0.90), xycoords='axes fraction', fontsize=16, fontweight='bold')


fig = plt.figure(figsize=(8, 7))
gs = GridSpec(2,1, figure=fig)

# Define plot locations
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])  # Row 0, Col 0-1


#print(np.argwhere(np.isnan(np.asarray(data))))

num_components = 2
test_for_mising_data(data,num_components,ax1,anno='A',load=True)
test_for_mising_data_vary_comp(data,ax2,anno='B',load=False)
#fig.tight_layout()
fig.savefig('C:/Users/patri/Coding/SystemsBiology__PPCA/figures/PPCA_MOFA.png')
plt.show()