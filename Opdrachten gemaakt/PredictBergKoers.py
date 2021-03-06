import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pylab as plt

import dmba

# load and preproces the data set
#big2021_df = pd.read_csv(r'C:\Users\marwi\Desktop\data\results\Results2021CatWT.csv')
riderInfos_df = pd.read_csv(r'C:\Users\marwi\Desktop\data\rider_infos.csv')
pd.set_option('max_columns', None)

# Create new columns derived from pps
list = []
for (i,r) in riderInfos_df.iterrows():
    e = r['pps']
    list.append(e)



listdict = []
for i in list:
    dict = eval(i)
    listdict.append(dict)


# Create dataframe where the dictionary values are stored
normalizedRiderInfos_df = pd.DataFrame(listdict)
normalizedRiderInfos_df.rename(columns={'One day races': 'One_day_races'}, inplace=True)


# Berg achtige etappe waar de kans groot is dat een klimmer of vluchter kan winnen
# Er is voor gekozen om de Amstel Gold race te voorspellen
koers = pd.DataFrame([{'Climber': 20000, 'One_day_races': 20000}])

# plot de koers en de gegeven renners van de dataset sprint op de x as, One day races op de y as.
# niet alle renners worden geplot dit omdat er een limiet op staat (nog naar kijken)
fig, ax = plt.subplots()


ax.scatter(normalizedRiderInfos_df.Climber, normalizedRiderInfos_df.One_day_races, marker='o', label='Renners', color='C1')

ax.scatter(koers.Climber, koers.One_day_races, marker='*', label='Koers', color='black')

plt.xlabel('Climber')  # set x-axis label
plt.ylabel('One_day_races')  # set y-axis label


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=4)

plt.show()

# fit scaler
scaler = preprocessing.StandardScaler()
scaler.fit(normalizedRiderInfos_df[['Climber', 'One_day_races']])

# Transform the full dataset
voorspelling = normalizedRiderInfos_df

# Use k-nearest neighbour
knn = NearestNeighbors(n_neighbors=15)
knn.fit(voorspelling[['Climber', 'One_day_races']])
distances, indices = knn.kneighbors(koers)
voorspelling.iloc[indices[0],:]  # indices bestaat uit een lijst van lijsten, wij hoeven alleen de eerste elementen te hebben

# print wielernaam
def drukAf():
    print("De top 15 renners die kans kans maken om een bergachtige koers te winnen: ")
    for i in indices[0]:
        print(riderInfos_df.iat[i,1])
drukAf()

