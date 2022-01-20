from __future__ import division
import os
import glob
from pandas.plotting import scatter_matrix
from typing import Union

import pandas as pd
import xgboost as xgb
from dmba import regressionSummary, exhaustive_search
import numpy as np
from pandas import DataFrame, Series
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from IPython.display import display
import statsmodels.api as sm
from mord import LogisticIT
import matplotlib.pylab as plt
import seaborn as sns
import dmba
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score
import pandas as pd
import numpy as np
import scipy.stats as scipy
import matplotlib.pyplot as plt
import requests
import io
import seaborn as sns

#import data into a big concated list
path = r'C:\Users\marwi\Desktop\data\results'                     # use your path
all_files = glob.iglob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
# doesn't create a list, nor does it append to one
#chris big mad

print(concatenated_df.columns)

#Cleaning and subsetting required data
ignore = ['Date', 'Start', 'Finish', 'Category', 'Race_url', 'Stage_url']
true_df = concatenated_df.drop(columns=ignore)
true_df.tail()

scatter_matrix(concatenated_df[['Rnk','GC','BiB','Age',]], figsize=(10,10))

# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = concatenated_df.drop(['Rnk'],1)
y_all = concatenated_df['Rnk']

# Standardising the data.
from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
cols = [['BiB','Age','UCI','Pnt']]
for col in cols:
    X_all[col] = scale(X_all[col])

def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)

        # Collect the revised columns
        output = output.join(col_data)

    return output
