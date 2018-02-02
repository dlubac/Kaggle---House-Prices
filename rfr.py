import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics


def preprocess(dataset):
    """
    Processes a dataset to factorize object features and impute missing values for numeric features

    :param dataset: a pandas dataframe read in from a file
    :return: processed dataframe
    """

    df = dataset
    
    # Factorize object (string) columns
    df_obj_cols = list(df.select_dtypes(include=[np.object]))
    df[df_obj_cols] = df[df_obj_cols].fillna('no_val')
    df[df_obj_cols] = df[df_obj_cols].apply(lambda x: pd.factorize(x)[0])
    
    # Impute missing values for numeric columns
    df_num_cols = list(df.select_dtypes(include=[np.number]))
    for col in df_num_cols:
        df[col] = df[col].replace(np.NaN, df[col].median())

    return df

# Read in and process datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train = preprocess(train)
test = preprocess(test)

# Apply log transformation to sale price
train['SalePrice'] = np.log(train['SalePrice'])

# Split training dataset
x = train[train.columns.values[1:-1]]
y = train[train.columns.values[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9320, test_size=0.33)

# Create a random forest model
rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
model = rfr.fit(x_train, y_train)
print('Model accuracy: ' + str(model.score(x_test, y_test)))

# Perform cross validation
'''cv_scores = cross_val_score(model, x, y, cv=10)
print('CV scores: ' + str(cv_scores))
print('CV mean: ' + str(cv_scores.mean()))'''

# Predict using test dataset
x_test_final = test[test.columns.values[1:]]
x_test_indicies = test[test.columns.values[0:1]]

predictions = model.predict(x_test_final)
predictions = pd.DataFrame(np.exp(predictions))

# Create submission file
filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_rfr.csv'
df_output = pd.concat([x_test_indicies, predictions], axis=1)
df_output.to_csv(filename, index=False, header=['Id', 'SalePrice'])
