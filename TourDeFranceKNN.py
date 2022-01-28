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


# split dataframe into train data and check data
trainData, validData = train_test_split(normalizedRiderInfos_df, test_size=0.4, random_state=26)
print(trainData.shape, validData.shape)

# Vlakke etappe waar vluchters en sprinters de beste mogelijkheid hebben om te winnen
# Er is voor gekozen om de Amstel Gold race te voorspellen
Stage_1 = pd.DataFrame([{'Sprint': 6000, 'One_day_races': 6000}])

#
def plotDataset(ax, data, showLabel=True, **kwargs):
    plt.xlabel('Sprint')  # set x-axis label
    plt.ylabel('One_day_races')  # set y-axis label

fig, ax = plt.subplots()

plotDataset(ax, trainData)
plotDataset(ax, validData, showLabel=False, facecolors='none')

ax.scatter(Stage_1.Sprint, Stage_1.One_day_races, marker='*', label='Stage 1', color='black', s=150)


plt.xlabel('Sprint')  # set x-axis label
plt.ylabel('One_day_races')  # set y-axis label
# for _, row in trainData.iterrows():
#    ax.annotate(row.Number, (row.Sprint + 2, row.Climber))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=4)

plt.show()

# fit scaler
scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['Sprint', 'One_day_races']])

# Transform the full dataset
# mowerNorm = pd.concat([pd.DataFrame(scaler.transform(mower_df[['Income', 'Lot_Size']]),
#                                    columns=['zIncome', 'zLot_Size']),
#                       mower_df[['Ownership', 'Number']]], axis=1)
trainNorm = normalizedRiderInfos_df.iloc[trainData.index]
validNorm = normalizedRiderInfos_df.iloc[validData.index]

# normalise stages
Stage_Norm = pd.concat([pd.DataFrame(scaler.transform(normalizedRiderInfos_df[['Sprint', 'One_day_races']]), columns=['zSprint', 'zOne_day_races'])])

# Use k-nearest neighbour
knn = NearestNeighbors(n_neighbors=15)
knn.fit(trainNorm[['zSprint', 'zOne_day_races']])
distances, indices = knn.kneighbors(Stage_Norm)
print(trainNorm.iloc[indices[0], :])  # indices is a list of lists, we are only interested in the first element

# Berg achtige etappe waar de kans groot is dat een klimmer of vluchter kan winnen
Stage_2 = pd.DataFrame([{'Climber': 6000, 'One_day_races': 6000}])

#
def plotDataset1(ax, data, showLabel=True, **kwargs):

    plt.xlabel('Climber')  # set x-axis label
    plt.ylabel('One_day_races')  # set y-axis label

fig, ax = plt.subplots()

plotDataset1(ax, trainData)
plotDataset1(ax, validData, showLabel=False, facecolors='none')

ax.scatter(Stage_2.Sprint, Stage_2.One_day_races, marker='*', label='Stage 1', color='black', s=150)


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=4)

plt.show()

# fit scaler
scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['Climber', 'One_day_races']])

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