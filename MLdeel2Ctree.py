import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt
import dmba
from dmba import plotDecisionTree, classificationSummary, regressionSummary

wine_df = dmba.load_data('Wine.csv')

classTree = DecisionTreeClassifier(random_state=0)
classTree.fit(wine_df.drop(columns=['Type']), wine_df['Type'])

print("Classes: {}".format(', '.join(classTree.classes_)))
plotDecisionTree(classTree, feature_names=wine_df.columns[:2], class_names=classTree.classes_)