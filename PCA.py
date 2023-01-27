#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[2]:


data = np.array([[2,3],[4,5],[6,5],[6,7],[7,8],[5,8]])
df = pd.DataFrame(data=data,columns=['X','Y'])
print('Our toy dataset:')
df


# # PCA

# ## Standardize the data

# In[3]:


from scipy.stats import zscore
df_scaled = df.apply(zscore)
print('Scaled Data:')
df_scaled


# ## Find the covariance matrix of the scaled data

# In[4]:


cov = np.cov(df_scaled,rowvar=False)
print('Covariance Matrix:')
print(cov)


# ### Fit the 2-dimensional data in PCA

# In[5]:


pca = PCA(n_components=2)
pca.fit(df_scaled)


# ### Find the Eigan Values

# In[6]:


print('Eigan Values:')
print(pca.explained_variance_)


# ### Find the Eigan Vectors

# In[7]:


print('Eigan Vectors:')
print(pca.components_)


# We have found eigan vectors for both the eigan values

# ### Project Data into new single dimension

# In[8]:


pca3 = PCA(n_components=1)
pca3.fit(df_scaled)
Xpca3 = pca3.transform(df_scaled)
print('Projected data:')
print(Xpca3)


# Our data is converted into single feature while retaining maximum information from previous features.
