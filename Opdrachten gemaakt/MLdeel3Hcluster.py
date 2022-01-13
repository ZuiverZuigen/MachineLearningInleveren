import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

import dmba

#load data and preprocess
pharma_df = dmba.load_data('Pharmaceuticals.csv')
pharma_df.set_index('Name', inplace=True)
ignore = ['Symbol', 'Median_Recommendation', 'Location', 'Exchange']
pharma_df = pharma_df.drop(columns=ignore)

# while not required, the conversion of integer data to float will avoid a warning when
# applying the scale function
pharma_df = pharma_df.apply(lambda x: x.astype('float64'))
pharma_df.head()

d = pairwise.pairwise_distances(pharma_df, metric='euclidean')
pd.DataFrame(d, columns=pharma_df.index, index=pharma_df.index).head(5)

# scikit-learn uses population standard deviation
pharma_df_norm = pharma_df.apply(preprocessing.scale, axis=0)

# pandas uses sample standard deviation
pharma_df_norm = (pharma_df - pharma_df.mean())/pharma_df.std()

# compute normalized distance based on market cap and profit margin
d_norm = pairwise.pairwise_distances(pharma_df_norm[['Market_Cap', 'Net_Profit_Margin']],
                                     metric='euclidean')
pd.DataFrame(d_norm, columns=pharma_df.index, index=pharma_df.index).head(5)

#Hierarchical Clustering Dendrogram (Single linkage)
Z = linkage(pharma_df_norm, method='single')
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(bottom=0.23)
plt.title('Hierarchical Clustering Dendrogram (Single linkage)')
plt.xlabel('Name')
dendrogram(Z, labels=pharma_df_norm.index, color_threshold=2.75)
plt.axhline(y=2.75, color='black', linewidth=0.5, linestyle='dashed')
plt.show()

#Hierarchical Clustering Dendrogram (Average linkage)
Z = linkage(pharma_df_norm, method='average')
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(bottom=0.23)
plt.title('Hierarchical Clustering Dendrogram (Average linkage)')
plt.xlabel('Name')
dendrogram(Z, labels=pharma_df_norm.index, color_threshold=3.6)
plt.axhline(y=3.6, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


#single linkage
memb = fcluster(linkage(pharma_df_norm, 'single'), 6, criterion='maxclust')
memb = pd.Series(memb, index=pharma_df_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))

#average linkage
memb = fcluster(linkage(pharma_df_norm, 'average'), 6, criterion='maxclust')
memb = pd.Series(memb, index=pharma_df_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


#hierachical cluster
pharma_df_norm.index = ['{}: {}'.format(cluster, state) for cluster, state in zip(memb, pharma_df_norm.index)]
sns.clustermap(pharma_df_norm, method='average', col_cluster=False,  cmap="mako_r")
plt.show()