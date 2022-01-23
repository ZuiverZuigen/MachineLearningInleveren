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
big2021_df = pd.read_csv (r'C:\Users\marwi\Desktop\data\results\Results2021CatWT.csv')
pd.set_option('max_columns', None)
#print(big_df)

#slice df tour de france
Tour2021_df = big2021_df.iloc[18397:22377]
print(Tour2021_df)

#Treat Stage_Name as categorical, convert to dummy variables
big_df['Stage_Name'] = big_df['Stage_Name'].astype('category')
new_categories = {1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Professional'}
bank_df.Education.cat.rename_categories(new_categories, inplace=True)
bank_df = pd.get_dummies(bank_df, prefix_sep='_', drop_first=True)