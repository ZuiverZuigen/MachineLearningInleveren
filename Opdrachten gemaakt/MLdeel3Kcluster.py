import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.pylab as fig
from pandas.plotting import parallel_coordinates

import dmba

# Load and preprocess data
pharma_df = dmba.load_data('Pharmaceuticals.csv')
pharma_df.set_index('Name', inplace=True)
ignore = ['Symbol', 'Median_Recommendation', 'Location', 'Exchange']
pharma_df = pharma_df.drop(columns=ignore)
pharma_df = pharma_df.apply(lambda x: x.astype('float64'))
pd.set_option('max_columns', None)


# Normalized distance
pharma_df_norm = pharma_df.apply(preprocessing.scale, axis=0)

kmeans = KMeans(n_clusters=6, random_state=0).fit(pharma_df_norm)

# Cluster membership
memb = pd.Series(kmeans.labels_, index=pharma_df_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


centroids = pd.DataFrame(kmeans.cluster_centers_, columns=pharma_df_norm.columns)
pd.set_option('precision', 3)
print(centroids)
pd.set_option('precision', 6)


withinClusterSS = [0] * 6
clusterCount = [0] * 6
for cluster, distance in zip(kmeans.labels_, kmeans.transform(pharma_df_norm)):
    withinClusterSS[cluster] += distance[cluster]**2
    clusterCount[cluster] += 1
for cluster, withClustSS in enumerate(withinClusterSS):
    print('Cluster {} ({} members): {:5.2f} within cluster'.format(cluster,
        clusterCount[cluster], withinClusterSS[cluster]))


# calculate the distances of each data point to the cluster centers
distances = kmeans.transform(pharma_df_norm)

# reduce to the minimum squared distance of each data point to the cluster centers
minSquaredDistances = distances.min(axis=1) ** 2

# combine with cluster labels into a data frame
df = pd.DataFrame({'squaredDistance': minSquaredDistances, 'cluster': kmeans.labels_},
                  index=pharma_df_norm.index)

# Group by cluster and print information
for cluster, data in df.groupby('cluster'):
    count = len(data)
    withinClustSS = data.squaredDistance.sum()
    print(f'Cluster {cluster} ({count} members): {withinClustSS:.2f} within cluster ')


centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]

plt.figure(figsize=(10,6))
fig.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
plt.xlim(0,30)
centroids
pharma_df_norm.groupby(kmeans.labels_).mean()

print(pd.DataFrame(pairwise.pairwise_distances(kmeans.cluster_centers_, metric='euclidean')))
print(pd.DataFrame(pairwise.pairwise_distances(kmeans.cluster_centers_, metric='euclidean')).sum(axis=0))

inertia = []
for n_clusters in range(1, 7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pharma_df_norm)
    inertia.append(kmeans.inertia_ / n_clusters)
inertias = pd.DataFrame({'n_clusters': range(1, 7), 'inertia': inertia})
ax = inertias.plot(x='n_clusters', y='inertia')
plt.xlabel('Number of clusters(k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.ylim((0, 1.1 * inertias.inertia.max()))
ax.legend().set_visible(False)
plt.show()