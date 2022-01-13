import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pylab as plt

import dmba

bank_df = dmba.load_data('UniversalBank.csv')
bank_df = bank_df.iloc[0:100]
trainData, validData =train_test_split(bank_df, test_size=0.4, random_state=20)

#new person
newPerson = pd.DataFrame([{'Age': 45, 'Income': 100}])

#scatter plot
def plotDataset(ax, data, showLabel=True, **kwargs):
    subset = data.loc[data['CreditCard'] == 1]
    ax.scatter(subset.Age, subset.Income, marker='o', label='Owner' if showLabel else None, color='C1', **kwargs)

    subset = data.loc[data['CreditCard'] == 0]
    ax.scatter(subset.Age, subset.Income, marker='D', label='Nonowner' if showLabel else None, color='C0',
               **kwargs)

    plt.xlabel('Age')  # set x-axis label
    plt.ylabel('Income')  # set y-axis label
    for _, row in data.iterrows():
        ax.annotate(row.ID, (row.Age + 2, row.Income))


fig, ax = plt.subplots()

plotDataset(ax, trainData)
plotDataset(ax, validData, showLabel=False, facecolors='none')

ax.scatter(newPerson.Age, newPerson.Income, marker='*', label='New person', color='black', s=150)

plt.xlabel('Age')  # set x-axis label
plt.ylabel('Income')  # set y-axis label

handles, labels = ax.get_legend_handles_labels()
ax.set_xlim(0, 100)
ax.legend(handles, labels, loc=4)

plt.show()

#normalising the data
predictors = ['Age', 'Income']
outcome = []

scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['Age', 'Income']])  # Note the use of an array of column names

# Transform the full dataset
bankNorm = pd.concat([pd.DataFrame(scaler.transform(bank_df[['Age', 'Income']]),
                                    columns=['zAge', 'zIncome']),
                       bank_df[['CreditCard', 'ID']]], axis=1)
trainNorm = bankNorm.iloc[trainData.index]
validNorm = bankNorm.iloc[validData.index]
newPersonNorm = pd.DataFrame(scaler.transform(newPerson), columns=['zAge', 'zIncome'])

#K-NN
knn = NearestNeighbors(n_neighbors=3)
knn.fit(trainNorm[['zAge', 'zIncome']])
distances, indices = knn.kneighbors(newPersonNorm)
print(trainNorm.iloc[indices[0], :])  # indices is a list of lists, we are only interested in the first element

#results table
train_X = trainNorm[['zAge', 'zIncome']]
train_y = trainNorm['CreditCard']
valid_X = validNorm[['zAge', 'zIncome']]
valid_y = validNorm['CreditCard']

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
bank_X = bankNorm[['zAge', 'zIncome']]
bank_y = bankNorm['CreditCard']
knn = KNeighborsClassifier(n_neighbors=4).fit(bank_X, bank_y)
distances, indices = knn.kneighbors(newPersonNorm)
print(knn.predict(newPersonNorm))
print('Distances',distances)
print('Indices', indices)
print(bankNorm.iloc[indices[0], :])