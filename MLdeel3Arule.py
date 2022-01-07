import heapq
from collections import defaultdict

import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

import dmba


# load dataset
all_cars_df = dmba.load_data('ToyotaCorolla.csv')

# create the binary incidence matrix
ignore = ['Id', 'Model', 'Price', 'Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM', 'Fuel_Type',
          'HP', 'Color', 'CC', 'Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight', 'Guarantee_Period']
type_cars = all_cars_df.drop(columns=ignore)
type_cars[type_cars > 0] = 1

type_cars.head()