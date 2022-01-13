import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
import statsmodels.formula.api as sm
import matplotlib.pylab as plt

import dmba
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

#Select columns for regression analysis
bank_df = dmba.load_data('UniversalBank.csv')
predictors = ['Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Personal Loan']

outcome = 'Age'

#Partition data
x = pd.get_dummies(bank_df[predictors], drop_first=True)
y = bank_df[outcome]
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.4, random_state=1)

bank_lm = LinearRegression()
bank_lm.fit(train_x, train_y)

#prediction
bank_lm_predict = bank_lm.predict(valid_x)

result = pd.DataFrame({'Predicted': bank_lm_predict, 'Actual': valid_y,
                       'Residual': valid_y - bank_lm_predict})

print(result.head(25))

#Regression statistics
regressionSummary(valid_y, bank_lm_predict)

pred_y = bank_lm.predict(train_x)
print('adjusted r2 : ', adjusted_r2_score(train_y, pred_y, bank_lm))
print('AIC : ', AIC_score(train_y, pred_y, bank_lm))
print('BIC : ', BIC_score(train_y, pred_y, bank_lm))

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

#histogram
bank_lm_pred = bank_lm.predict(valid_x)
all_residuals = valid_y - bank_lm_pred

# Determine the percentage of datapoints with a residual in [-1406, 1406] = approx. 75\%
print(len(all_residuals[(all_residuals > -1406) & (all_residuals < 1406)]) / len(all_residuals))

ax = pd.DataFrame({'Residuals': all_residuals}).hist(bins=25)

plt.tight_layout()
plt.show()