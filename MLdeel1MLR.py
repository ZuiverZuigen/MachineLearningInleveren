import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import dmba
from dmba import regressionSummary

#Select columns for regression analysis
wine_df = dmba.load_data('Wine.csv')

predictors = ['Type', 'Alcohol', 'Ash', 'Ash_Alcalinity', 'Magnesium', 'Total_Phenols',
              'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280_OD315', 'Proline']

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