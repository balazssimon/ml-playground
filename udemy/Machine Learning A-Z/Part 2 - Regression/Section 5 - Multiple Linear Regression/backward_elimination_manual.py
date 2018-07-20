# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# One hot encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray() 

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# The Linear Regression library includes the constant, but the statsmodels does not
# so we have to add it to our model:
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)

# Building the optimal model using Backwards Elimination
import statsmodels.formula.api as sm
# Step 1
SL = 0.05
# Step 2, using Ordinary Least Squares from statsmodels (instead of Linear Regression from linear_model)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3
regressor_OLS.summary()

# Step 4
X_opt = X[:,[0,1,3,4,5]]
# Step 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3
regressor_OLS.summary()

# Step 4
X_opt = X[:,[0,3,4,5]]
# Step 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3
regressor_OLS.summary()

# Step 4
X_opt = X[:,[0,3,5]]
# Step 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3
regressor_OLS.summary()

# Step 4
X_opt = X[:,[0,3]]
# Step 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3
regressor_OLS.summary()
# Finished
