import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.preprocessing import Imputer

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

# Replace NAs in numeric columns with mean
#numeric_cols = list(train.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']))
#train[numeric_cols] = train[numeric_cols].apply(lambda x: x.fillna(np.rint(x.mean())))

# Remove any columns in NAs
cols_with_na = train.columns[train.isnull().any()].tolist()
train = train.drop(cols_with_na, axis=1)

# Factorize object (string) columns
df_obj_cols = list(train.select_dtypes(include=[np.object]))
train[df_obj_cols] = train[df_obj_cols].apply(lambda x: pd.factorize(x)[0])

# Impute missing values
imp = Imputer(missing_values=-1, strategy='most_frequent', copy=False)
train_imputed = imp.fit_transform(train)
train = pd.DataFrame(train_imputed, index=train.index, columns=train.columns)

# Split training dataset
x = train[train.columns.values[1:-1]]
y = train[train.columns.values[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9320, test_size=0.33)

# Create a random forest model
rfr = RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, n_jobs=-1)
model = rfr.fit(x_train, y_train)
print('Model accuracy: ' + str(model.score(x_test, y_test)))

# Visualize model accuracy
predictions = model.predict(x_test)
sns.regplot(predictions, y_test, marker='.')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Random Forest Model Accuracy')
#plt.show()

