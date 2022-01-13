import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pylab as plt

import dmba

wine_df = dmba.load_data('Wine.csv')
trainData, validData = train_test_split(wine_df,
                                        test_size=0.4, random_state=1)
#normaliseer
scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['Malic_Acid', 'Flavanoids']])

#transform
wineNorm = pd.concat([pd.DataFrame(scaler.transform
                                   (wine_df[['Malic_Acid', 'Flavanoids']]),
                                   columns=['zMalic_Acid', 'zFlavanoids'])])

trainNorm = wineNorm.iloc[trainData.index]
validNorm = wineNorm.iloc[validData.index]

knn = NearestNeighbors(n_neighbors=3)
knn.fit(trainNorm[['zMalic_Acid', 'zFlavanoids']])
distances, indices = knn.kneighbors(wineNorm)
print(trainNorm.iloc[indices[0], :])