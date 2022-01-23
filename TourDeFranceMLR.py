import heapq
from collections import defaultdict

import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

import dmba

#load and preproces the data set
big_df = pd.read_csv (r'C:\Users\marwi\Desktop\data\results\Results2021CatWT.csv')
pd.set_option('max_columns', None)
#print(big_df)

#UAE Tour stage 1
df_UAE = big_df.iloc[:139]
print(df_UAE)