import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pylab as plt

import dmba

cereals_df = dmba.load_data('Cereals.csv')
cereals_df['number'] = cereals_df.index + 1
trainData, validData = train_test_split(cereals_df, test_size=0.4, random_state=26)


#new cereal
newCereal = pd.DataFrame([{'calories': 85, 'rating': 65.0}])


#scatter plot
def plotDataset(ax, data, showLabel=True, **kwargs):
    ax.scatter(cereals_df.calories, cereals_df.rating, marker='o', color='C1', **kwargs)
    plt.xlabel('calories')  # set x-axis label
    plt.ylabel('rating')  # set y-axis label
    for _, row in data.iterrows():
        ax.annotate(row.number, (row.calories + 2, row.rating))


fig, ax = plt.subplots()

plotDataset(ax, trainData)
plotDataset(ax, validData, showLabel=False, facecolors='none')

ax.scatter(newCereal.calories, newCereal.rating, marker='*', label='New cereal', color='black', s=150)

plt.xlabel('Calories')  # set x-axis label
plt.ylabel('Rating')  # set y-axis label

handles, labels = ax.get_legend_handles_labels()
ax.set_xlim(40, 115)
ax.legend(handles, labels, loc=4)

plt.show()


scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['calories', 'rating']])  # Note the use of an array of column names

# Transform the full dataset
cerealNorm = pd.concat([pd.DataFrame(scaler.transform(cereals_df[['calories', 'rating']]),
                                    columns=['zCalories', 'zRating']),
                       cereals_df[['name', 'number']]], axis=1)

trainNorm = cerealNorm.iloc[trainData.index]
validNorm = cerealNorm.iloc[validData.index]
newCerealNorm = pd.DataFrame(scaler.transform(newCereal), columns=['zCalories', 'zRating'])

knn = NearestNeighbors(n_neighbors=3)
knn.fit(trainNorm[['zCalories', 'zRating']])
distances, indices = knn.kneighbors(newCerealNorm)
print(trainNorm.iloc[indices[0], :])  # indices is a list of lists, we are only interested in the first element


train_X = trainNorm[['zCalories', 'zRating']]
train_y = trainNorm['name']
valid_X = validNorm[['zCalories', 'zRating']]
valid_y = validNorm['name']

# Train a classifier for different values of k
results = []
for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=k).fit(train_X, train_y)
    results.append({
        'k': k,
        'accuracy': accuracy_score(valid_y, knn.predict(valid_X))
    })

# Convert results to a pandas data frame
results = pd.DataFrame(results)
print(results)

# Retrain with full dataset
cereal_X = cerealNorm[['zCalories', 'zRating']]
cereal_y = cerealNorm['name']
knn = KNeighborsClassifier(n_neighbors=4).fit(cereal_X, cereal_y)
distances, indices = knn.kneighbors(newCerealNorm)
print(knn.predict(newCerealNorm))
print('Distances',distances)
print('Indices', indices)
print(cerealNorm.iloc[indices[0], :])