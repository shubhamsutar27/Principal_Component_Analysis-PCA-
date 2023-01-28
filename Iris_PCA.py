#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[67]:


df = pd.read_csv('Iris.csv')
df.head()


# In[68]:


df.drop('Id',axis=1,inplace=True)
df.head()


# In[69]:


x = df.iloc[:,0:4].values
y = df.iloc[:,4].values


# In[70]:


x


# In[71]:


y


# ## Standardization

# In[72]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[73]:


x_scaled


# ## Finding Covariance matrix

# In[74]:


#con_matrix = np.cov(x_scaled.T)


# In[75]:


# Initialize the PCA object
pca = PCA()

#fit and transform the data
df_pca = pca.fit_transform(x_scaled)

#compute the covariance matrix
con_matrix = pca.get_covariance()


# In[76]:


con_matrix


# ## Finding Eigan values and Eigan vectors

# In[77]:


# Eigan values
eiganvalues = pca.explained_variance_

# Eigan vectors
eiganvectors = pca.components_


# In[78]:


eiganvalues


# In[79]:


eiganvectors


# ## Sorting Eigan values and corresponding Eigan vectors

# In[80]:


# Sort the eigenvalues in descending order
eiganvalues_sorted = sorted(eiganvalues, reverse=True)

# Get the indices of the sorted eigenvalues
eiganvalues_indices = np.argsort(-eiganvalues)


# In[81]:


eiganvalues_sorted


# ## Finding optimum principal components 

# In[82]:


# Plot the scree plot
plt.plot(eiganvalues)
plt.xlabel('Number of principal components')
plt.ylabel('Eiganvalues')
plt.show()


# From the plot given above , 2 are the optimum number of principal components

# In[83]:


df2 = PCA(n_components=2)
df2.fit(x_scaled)
data = df2.transform(x_scaled)
print('Projected data:')
print(data)


# In[84]:


data.dtype


# In[85]:


data1 = pd.DataFrame(data,columns = ['pc1','pc2'])
data1.head()


# In[86]:


finalDf = pd.concat([data1,pd.DataFrame(y,columns = ['species'])], axis = 1)
finalDf.head()


# In[87]:


import seaborn as sns

sns.heatmap(pca.components_, cmap='coolwarm', annot=True)
plt.xlabel('Original Feature')
plt.ylabel('Principal Component')
plt.show()


# In[88]:


import matplotlib.pyplot as plt

plt.scatter(finalDf.iloc[:,0], finalDf.iloc[:,1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[89]:


import seaborn as sns

sns.scatterplot(data=df)


# In[90]:


import seaborn as sns

sns.scatterplot(data=finalDf,x='pc1',y='pc2',hue='species')


# From the scatter plots given above for 'with PCA 'and 'without PCA' , we can see that with pca, the plot looks clean and easy to interprete.
