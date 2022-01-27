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
# Create dataframe where you'll store the dictionary values
list = []
for (i,r) in riderInfos_df.iterrows():
    e = r['pps']
    list.append(e)



listdict = []
for i in list:
    dict = eval(i)
    listdict.append(dict)



normalizedRiderInfos_df = pd.DataFrame(listdict)
normalizedRiderInfos_df.head(20)

#print(big2021_df)
# vraag over hoe je dit nou precies doet
#riderInfos_df = riderInfos_df["pps"].str.replace('GC:', '')
#normalizedRiderInfos_df = pd.read_csv(r'C:\Users\marwi\Desktop\data\rider_infosManuela.csv')
#normalizedRiderInfos_df = normalizedRiderInfos_df.loc[:5]
#print(riderInfos_df.head(20))

# slice df tour de france
#Tour2021_df = big2021_df.loc[18397:22377]

# split dataframe into train data and check data
trainData, validData = train_test_split(normalizedRiderInfos_df, test_size=0.4, random_state=26)
print(trainData.shape, validData.shape)

# Vlakke etappe waar vluchters en sprinters de beste mogelijkheid hebben om te winnen
# Er is voor gekozen om de Amstel Gold race te voorspellen
Stage_1 = pd.DataFrame([{'Sprint': 6000, 'One day races': 6000}])

#
def plotDataset(ax, data, showLabel=True, **kwargs):
 # subset = data.loc[data['Ownership']=='Owner']
 # ax.scatter(subset.Income, subset.Lot_Size, marker='o', label='Owner' if showLabel else None, color='C1', **kwargs)

  # subset = data.loc[data['Ownership']=='Nonowner']
    # ax.scatter(subset.Income, subset.Lot_Size, marker='D', label='Nonowner' if showLabel else None, color='C0', **kwargs)

    plt.xlabel('Sprint')  # set x-axis label
    plt.ylabel('One day races')  # set y-axis label
#    for _, row in data.iterrows():
#        ax.annotate(row.Number, (row.Income + 2, row.Lot_Size))
fig, ax = plt.subplots()

plotDataset(ax, trainData)
plotDataset(ax, validData, showLabel=False, facecolors='none')

ax.scatter(Stage_1.Sprint, Stage_1.ODR, marker='*', label='Stage 1', color='black', s=150)


plt.xlabel('Sprint')  # set x-axis label
plt.ylabel('One day races')  # set y-axis label
# for _, row in trainData.iterrows():
#    ax.annotate(row.Number, (row.Sprint + 2, row.Climber))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=4)

plt.show()

# fit scaler
scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['Sprint', 'One day races']])

# Transform the full dataset
# mowerNorm = pd.concat([pd.DataFrame(scaler.transform(mower_df[['Income', 'Lot_Size']]),
#                                    columns=['zIncome', 'zLot_Size']),
#                       mower_df[['Ownership', 'Number']]], axis=1)
trainNorm = normalizedRiderInfos_df.iloc[trainData.index]
validNorm = normalizedRiderInfos_df.iloc[validData.index]

# normalise stages
Stage_Norm = pd.DataFrame(scaler.transform(Stage_1), columns=['zSprint', 'zOne_day_races'])

# Use k-nearest neighbour
knn = NearestNeighbors(n_neighbors=15)
knn = NearestNeighbors(n_neighbors=3)
knn.fit(trainNorm[['zSprint', 'zOne_day_races']])
distances, indices = knn.kneighbors(Stage_Norm)
print(trainNorm.iloc[indices[0], :])  # indices is a list of lists, we are only interested in the first element

# Berg achtige etappe waar de kans groot is dat een klimmer of vluchter kan winnen
Stage_2 = pd.DataFrame([{'Climber': 6000, 'One day races': 6000}])

#
def plotDataset(ax, data, showLabel=True, **kwargs):
 #   subset = data.loc[data['Ownership']=='Owner']
 #   ax.scatter(subset.Income, subset.Lot_Size, marker='o', label='Owner' if showLabel else None, color='C1', **kwargs)

  #  subset = data.loc[data['Ownership']=='Nonowner']
    # ax.scatter(subset.Income, subset.Lot_Size, marker='D', label='Nonowner' if showLabel else None, color='C0', **kwargs)

    plt.xlabel('Climber')  # set x-axis label
    plt.ylabel('One day races')  # set y-axis label
#    for _, row in data.iterrows():
#        ax.annotate(row.Number, (row.Income + 2, row.Lot_Size))
fig, ax = plt.subplots()

plotDataset(ax, trainData)
plotDataset(ax, validData, showLabel=False, facecolors='none')

ax.scatter(Stage_2.Sprint, Stage_2.ODR, marker='*', label='Stage 1', color='black', s=150)


plt.xlabel('Climber')  # set x-axis label
plt.ylabel('One day races')  # set y-axis label
# for _, row in trainData.iterrows():
#    ax.annotate(row.Number, (row.Sprint + 2, row.Climber))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=4)

plt.show()

# fit scaler
scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['Climber', 'One day races']])

# Transform the full dataset
# mowerNorm = pd.concat([pd.DataFrame(scaler.transform(mower_df[['Income', 'Lot_Size']]),
#                                    columns=['zIncome', 'zLot_Size']),
#                       mower_df[['Ownership', 'Number']]], axis=1)
trainNorm1 = normalizedRiderInfos_df.iloc[trainData.index]
validNorm1 = normalizedRiderInfos_df.iloc[validData.index]

# normalise stages
Stage_Norm1 = pd.DataFrame(scaler.transform(Stage_1), columns=['zClimber', 'zOne_day_races'])

# Use k-nearest neighbour
knn = NearestNeighbors(n_neighbors=15)
knn = NearestNeighbors(n_neighbors=3)
knn.fit(trainNorm[['zClimber', 'zOne_day_races']])
distances, indices = knn.kneighbors(Stage_Norm1)
print(trainNorm.iloc[indices[0], :])  # indices is a list of lists, we are only interested in the first element