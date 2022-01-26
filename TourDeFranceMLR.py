import heapq
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
import statsmodels.formula.api as sm
import matplotlib.pylab as plt

import dmba
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score


#load and preproces the data set
big2021_df = pd.read_csv (r'C:\Users\marwi\Desktop\data\results\Results2021CatWT.csv')
riderInfos_df = pd.read_csv(r'C:\Users\marwi\Desktop\data\rider_infos.csv')
pd.set_option('max_columns', None)
#print(big_df)

riderInfos_df["pps"].str.replace(0, '')
print(riderInfos_df.pps.head(20))

#slice df tour de france
Tour2021_df = big2021_df.loc[18397:22377]

predictors = ['Rnk', 'Age', 'Stage_Type', 'Stage#']
outcome = ''

# partition data
X = pd.get_dummies(Tour2021_df[predictors], drop_first=True)
y = Tour2021_df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

tour_lm = LinearRegression()
tour_lm.fit(train_X, train_y)

pred_y = tour_lm.predict(train_X)

# Use predict() to make predictions on a new set
tour_lm_pred = tour_lm.predict(valid_X)

result = pd.DataFrame({'Predicted': tour_lm_pred, 'Actual': valid_y})
print(result.head(20))

# Compute common accuracy measures
regressionSummary(valid_y, tour_lm_pred)