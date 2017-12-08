import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# Set seed for reproducability
np.random.seed(9320)

# Read in data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Basic analysis on price to check for normality
prices = train['SalePrice']
sns.distplot(prices, fit=norm)
#plt.show()

# Apply log transformation to price
train['SalePrice'] = np.log(train['SalePrice'])
prices_log = train['SalePrice']
sns.distplot(prices_log, fit=norm)
#plt.show()

# Find columns with NaN values
cols_with_na = train.columns[train.isnull().any()].tolist()

# Remove columns with NaN values
train = train.drop(cols_with_na, axis=1)

# Factorize object (string) columns
df_obj_cols = list(train.select_dtypes(include=[np.object]))
train[df_obj_cols] = train[df_obj_cols].apply(lambda x: pd.factorize(x)[0])

# Split training dataset
msk = np.random.rand(len(train)) < 0.7
df_train = train[msk]
df_test = train[~msk]

x_train = df_train[df_train.columns.values[1:-1]]
y_train = df_train[df_train.columns.values[-1]]

x_test = df_train[df_test.columns.values[1:-1]]
y_test = df_train[df_test.columns.values[-1]]

# Create random forest regressor and model
rfr = RandomForestRegressor(n_estimators=64, n_jobs=-1)
model = rfr.fit(x_train, y_train)

# View most important features to model
coef = pd.Series(rfr.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print(coef.head(25))

# Predict values
y_preds = rfr.predict(x_test)

print(len(df_train))
print(len(df_test))
print(len(y_preds))


