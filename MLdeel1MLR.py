import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import dmba
from dmba import regressionSummary
from dmba import backward_elimination
from dmba import AIC_score

#Select columns for regression analysis
wine_df = dmba.load_data('Wine.csv')

predictors = ['Nonflavanoid_Phenols', 'Hue']

outcome = 'Malic_Acid'

#Partition data
x = pd.get_dummies(wine_df[predictors], drop_first=True)
y = wine_df[outcome]
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=1)

wine_lm = LinearRegression()
wine_lm.fit(train_x, train_y)

#prediction
wine_lm_pred = wine_lm.predict(valid_x)

result = pd.DataFrame({'Predicted': wine_lm_pred, 'Actual': valid_y,
                       'Residual': valid_y - wine_lm_pred})

print(result.head(20))

#Regression statistics
regressionSummary(valid_y, wine_lm_pred)

#Backward elimination
def train_model(variables):
    model = LinearRegression()
    model.fit(train_x[variables], train_y)
    return model

def score_model(model, variables):
    return AIC_score(train_y, model.predict
    (train_x[variables]), model)

best_model, best_variables = backward_elimination(train_x.columns,
                                                  train_model, score_model, verbose=True)

print(best_variables)