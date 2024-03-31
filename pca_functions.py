import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from scipy.linalg import orth
from sklearn.decomposition import PCA


# Principal component analysis function 

def pca(data, num_comp): #data, number of components
    miss_ids = np.argwhere(np.isnan(np.asarray(data)))

    data_norm = (data - np.nanmean(np.asarray(data), axis=0)) # normalize the data
    if len(miss_ids)>0:
        for m in miss_ids:
            data_norm[m[0],m[1]]=0
    cov_mat = np.cov(data_norm.T) #covariance matrix

    eig_vals, eig_vecs = np.linalg.eigh(cov_mat) #eigenvalues and eigenvectors of covariance matrix

    # Sort eigenvectors in descending order of eigenvalues
    indices = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, indices]
    eig_vals = eig_vals[indices]
    # Select the top eigenvectors and eigenvalues
    eig_vecs_selected = eig_vecs[:, :num_comp]
    eig_vals_selected = eig_vals[ :num_comp]

    # Project the data onto the new subspace defined by the selected eigenvectors
    pca_data = data_norm @ eig_vecs_selected
    recon_data_pca = np.asarray((pca_data @ eig_vecs_selected.T) + np.nanmean(np.asarray(data), axis=0))

    return  pca_data, eig_vecs_selected, eig_vals_selected, recon_data_pca




# probabilistic principal component analysis function

def ppca(t, num_comp,W_init,sigma_init, max_iter=1000, tol=0.0001): # dataset, number of components, initial W, initial sigma^2
    n_samples, n_features = t.shape
    miss_ids = np.argwhere(np.isnan(np.asarray(t)))

    t_norm = np.asarray(t - np.nanmean(np.asarray(t), axis=0)) #normalize the data
    if len(miss_ids)>0:
        for m in miss_ids:
            t_norm[m[0],m[1]] = 0

    W = W_init     # initialise the W-matrix and sigma^2
    s_squared = sigma_init
    recon_data_ppca=t_norm


    for i in range(max_iter):  # iterate the E and M steps until convergence or max_iter is reached or 
        S=np.cov(t_norm.T) # calculate the covariance matrix of the dataset

        M = (W.T @ W) + s_squared * np.eye(num_comp)  
        
        #E_xt = np.linalg.inv(M) @ W.T @ t_norm.T 
        
        #recon_data_ppca = (W @ np.linalg.inv(W.T @ W) @ M @ E_xt).T
        if len(miss_ids)>0:
            for m in miss_ids:
                t_norm[m[0],m[1]] = recon_data_ppca[m[0],m[1]]
        E_xt =  t_norm @ W @ np.linalg.inv(M)
        recon_data_ppca = E_xt @ W.T

        MWSW = np.linalg.inv(M) @ W.T @ S @ W 
        W_new = S @ W @ np.linalg.inv(s_squared * np.eye(num_comp)+MWSW)
        
        SWMW = S @ W @ np.linalg.inv(M) @ W_new.T
        #print(s_squared * len(miss_ids)/n_features)

        #s_squared_new = 1 / (n_samples * n_features) * (n_samples * s_squared * np.trace(W @ np.linalg.inv(M) @ W.T) + np.sum((E_xt @ W.T - t_norm)**2) + len(miss_ids) * s_squared)

        if len(miss_ids)>0:
            s_squared_new = np.trace(S - SWMW)/n_features + s_squared * len(miss_ids)/(n_features * n_samples)
        else:
            s_squared_new = np.trace(S - SWMW)/n_features
        
        # Check for convergence 
        if (np.abs(s_squared_new-s_squared) < tol):
            iter=i+1
            break
        if i+1==max_iter:
            iter=i+1
            break

        W=W_new
        s_squared=s_squared_new

    #E_xt = np.linalg.inv(M) @ W.T @ t_norm.T
    #print(t_norm[0:5,0:5])
    #print(recon_data_ppca[0:5,0:5])
    #recon_data_ppca = (W @ np.linalg.inv(W.T @ W) @ M @ E_xt)
    recon_data_ppca = [recon_data_ppca[n]+np.nanmean(np.asarray(t), axis=0) for n in range(len(recon_data_ppca))]
    E_xt=E_xt.T

    ppca_data = np.dot(t_norm, W) # Project the data onto the new subspace defined by the eigenvectors
    return ppca_data, W, s_squared, E_xt, recon_data_ppca, iter

def RMSE(original_data,recon_data):
    # Calculate the squared differences between corresponding data points
    squared_diff = [(np.array(og) - np.array(rec)) ** 2 
                    for og, rec in zip(original_data, recon_data)]
    # Calculate the mean of the squared differences
    mse = (np.mean(squared_diff))**(1/2)

    return mse