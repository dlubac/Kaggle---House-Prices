import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Read in data
train = pd.read_csv('data/train.csv')

# First look at data
print(train.head())

# Shape of data
print(str(train.shape))

# Check data types
for column in train:
    name = train[column].name
    dtype = train[column].dtype
    sample = train[column].sample(5).tolist()

    print(name + ': ' + str(dtype))
    print(sample)
    print('\n')

# Check percentage of nulls
nulls_list = []

for column in train:
    dt = {'name': train[column].name, 'percentage_null': train[column].isnull().sum() / len(train[column])}
    nulls_list.append(dt)

df_nulls = pd.DataFrame(nulls_list)
df_nulls = df_nulls[df_nulls['percentage_null'] > 0]
df_nulls = df_nulls.sort_values('percentage_null', ascending=False)

print(df_nulls)

# Summarize numerical features
print(train.describe())

# Visualize sale prices
plt.hist(train['SalePrice'], bins=20, edgecolor='black', linewidth=0.5)
plt.title('Sale Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix of numerical features
corr = train.select_dtypes(include=['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr, square=True, cmap="Blues")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

corr_list = []

for index, row in corr[corr.columns[-1]].iteritems():
    dt = {'feature': index, 'correlation': row}
    corr_list.append(dt)

df_corr = pd.DataFrame(corr_list, columns=('feature', 'correlation'))
df_corr = df_corr[df_corr['feature'] != 'SalePrice']
df_corr = df_corr.sort_values('correlation', ascending=False)

print(df_corr)

# Visualizing highly correlated features
sns.regplot(x='OverallQual', y='SalePrice', data=train)
plt.show()

# Visualize several other highly correlated features
fig, axs = plt.subplots(figsize=(10, 15), nrows=4, ncols=2)

ax1 = sns.regplot(x='GrLivArea', y='SalePrice', data=train, marker='.', ax=axs[0][0])
ax1.set_title('GrLivArea', fontdict={'fontsize': 10})

ax2 = sns.regplot(x='1stFlrSF', y='SalePrice', data=train, marker='.', ax=axs[0][1])
ax2.set_title('1stFlrSF', fontdict={'fontsize': 10})

ax3 = sns.regplot(x='GarageArea', y='SalePrice', data=train, marker='.', ax=axs[1][0])
ax3.set_title('GarageArea', fontdict={'fontsize': 10})

ax4 = sns.regplot(x='GarageCars', y='SalePrice', data=train, marker='.', ax=axs[1][1])
ax4.set_title('GarageCars', fontdict={'fontsize': 10})

ax5 = sns.regplot(x='FullBath', y='SalePrice', data=train, marker='.', ax=axs[2][0])
ax5.set_title('FullBath', fontdict={'fontsize': 10})

ax6 = sns.regplot(x='TotRmsAbvGrd', y='SalePrice', data=train, marker='.', ax=axs[2][1])
ax6.set_title('TotRmsAbvGrd', fontdict={'fontsize': 10})

ax7 = sns.regplot(x='YearBuilt', y='SalePrice', data=train, marker='.', ax=axs[3][0])
ax7.set_title('YearBuilt', fontdict={'fontsize': 10})

ax8 = sns.regplot(x='YearRemodAdd', y='SalePrice', data=train, marker='.', ax=axs[3][1])
ax8.set_title('YearRemodAdd', fontdict={'fontsize': 10})

plt.show()

# Categorical feature analysis
print(train.select_dtypes(include=['object']).columns.values)

sns.boxplot(x='Neighborhood', y='SalePrice', data=train)
sns.boxplot(x='BldgType', y='SalePrice', data=train)
sns.boxplot(x='HouseStyle', y='SalePrice', data=train)
sns.boxplot(x='RoofMatl', y='SalePrice', data=train)
sns.boxplot(x='ExterCond', y='SalePrice', data=train)
sns.boxplot(x='KitchenQual', y='SalePrice', data=train)
sns.boxplot(x='PavedDrive', y='SalePrice', data=train)
plt.xticks(rotation=90)
plt.show()
