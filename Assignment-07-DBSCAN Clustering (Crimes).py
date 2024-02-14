#!/usr/bin/env python
# coding: utf-8

# # Assignment-07-DBSCAN Clustering (Crimes)

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 300
import seaborn as sns
sns.set_theme(style='darkgrid',palette='viridis')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


# In[2]:


# Import Dataset
crime=pd.read_csv('crime_data.csv')
crime


# In[3]:


crime.info()


# In[4]:


crime.drop(['Unnamed: 0'],axis=1,inplace=True)
crime


# In[5]:


# Normalize heterogenous numerical data using standard scalar fit transform to dataset
x=StandardScaler().fit_transform(crime)
x


# # Model Building 

# In[6]:


# DBSCAN Clustering
db=DBSCAN(eps=1,min_samples=4)
y=db.fit_predict(crime)


# In[7]:


#Noisy samples are given the label -1.
db.labels_


# In[8]:


# Adding clusters to dataset
crime['clusters']=db.labels_
crime


# In[9]:


crime.groupby('clusters').agg(['mean']).reset_index()


# In[10]:


# Plot Clusters 
plt.scatter(crime['clusters'],crime['UrbanPop'], c=db.labels_,cmap='rainbow') 


# In[11]:


from sklearn.metrics import silhouette_score


# In[12]:


silhouette_score


# In[13]:


# k-means algorithm
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
k_means = kmeans.fit_predict(crime)


# In[14]:


k_means


# In[15]:


k_means1=k_means+1
k_cluster = list(k_means1)


# In[16]:


crime['k_cluster'] = k_cluster


# In[17]:


kmeans_mean_cluster = pd.DataFrame(round(crime.groupby('k_cluster').mean(),1))


# In[18]:


kmeans_mean_cluster


# In[19]:


pd.DataFrame(round(crime.groupby('k_cluster').count(),1))


# In[20]:


X = crime.values


# In[21]:


plt.scatter(X[:, 0], X[:, 1], c=k_means, s=50, cmap='viridis')


# In[22]:


# herarkical clustering
dendrogram = sch.dendrogram(sch.linkage(crime, method='ward'))


# In[23]:


model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')


# In[24]:


h_cluster = model.fit(X)


# In[25]:


labels = model.labels_


# In[26]:


plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')


# In[ ]:




