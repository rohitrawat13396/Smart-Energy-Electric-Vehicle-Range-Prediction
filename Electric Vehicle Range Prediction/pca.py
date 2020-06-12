from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("processed.csv", index_col='epoch time')

def pca(dataset):
 
    pca = PCA(n_components=26)
    pcomponent = pca.fit_transform(dataset)
    exp_var = pca.explained_variance_ratio_
    cum_exp_var = np.cumsum(pca.explained_variance_ratio_)
    principalDf = pd.DataFrame(pcomponent)
    pcomponent = np.transpose(principalDf).values.tolist()    
    return pcomponent,  exp_var.tolist(),   cum_exp_var.tolist(),   pca

pcomponent_mm,  expl_var_ratio_mm,  cum_expl_var_ratio_mm, pca_mm = pca(dataset)

plt.bar(range(0,26),expl_var_ratio_mm)
plt.plot(range(0,26),cum_expl_var_ratio_mm)
plt.ylabel("Explained Variance")
plt.xlabel("PCA Components")
plt.title("Explained Variance of different PCA Components")
plt.savefig("scree_plot")
plt.show()


def top5_features(pca,colnames):
    loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    df_loading  = pd.DataFrame(loadings,index=colnames)
    df_loading["sum_loadings"]= np.sum(df_loading,axis=1)
    df_loading = df_loading.sort_values(by = ["sum_loadings"],ascending= False)
    return list(df_loading.index[0:5]),df_loading

colnames = list(dataset.columns)
top3_attr_mm, df    =   top5_features(pca_mm,colnames)

plt.bar(df.index[0:5], df.sum_loadings[0:5])
plt.xlabel("Feature")
plt.ylabel("PCA loading Score")
plt.title("Scoring importance of top 5 features")
plt.savefig("pca_loadings")
plt.show()

