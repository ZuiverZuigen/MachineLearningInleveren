import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pylab as plt

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
normalizedRiderInfos_df = normalizedRiderInfos_df.rename(columns={'One day races': 'One_day_races'})
print(normalizedRiderInfos_df.head(20))