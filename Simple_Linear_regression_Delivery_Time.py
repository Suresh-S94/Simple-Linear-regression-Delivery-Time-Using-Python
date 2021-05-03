import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew

data1 = pd.read_csv("E:\\Data Science_Excelr\\Simple Linear Regression\\delivery_time.csv")
data1

data1.describe()

data1.shape

data1 = data1.rename(columns = {'Delivery Time': 'DT', 'Sorting Time': 'ST'}, inplace = False)

data1.info()

print(kurtosis(data1.DT))
print(kurtosis(data1.ST))


print(skew(data1.DT))
print(skew(data1.ST))

import seaborn as sns
sns.pairplot(data1)

import seaborn as sn
import matplotlib.pyplot as plt
corrMatrix = data1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

import seaborn as sns
cols = data1.columns 
colours = ['#ffc0cb', '#ffff00']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours))

data1.boxplot(column=['DT'])

data1.boxplot(column=['ST'])

data1[data1.duplicated()].shape

data1['ST'].hist()

data1.boxplot(column=['ST'])

data1['ST'].value_counts().plot.bar()

sns.pairplot(data1)

data1.corr()

sns.distplot(data1['DT'])

sns.distplot(data1['ST'])

data1 = data1.rename(columns = {'Delivery Time': 'DT', 'Sorting Time': 'ST'}, inplace = False)

data1

import statsmodels.formula.api as smf
model = smf.ols("DT~ST",data = data1).fit()

sns.regplot(x="ST", y="DT", data=data1);

model.params

print(model.tvalues, '\n', model.pvalues)

(model.rsquared,model.rsquared_adj)

model.summary()

data_1=data1
data_1['DT'] = np.log(data_1['DT'])
data_1['ST'] = np.log(data_1['ST'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()

model_2 = smf.ols("ST~DT",data = data_1).fit()

model_2.summary()

data_2=data1
data_1['DT'] = np.log(data_1['DT'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()

model_3 = smf.ols("ST~DT",data = data_2).fit()

model_3.summary()

data_3=data1
data_1['ST'] = np.log(data_1['ST'])
sns.distplot(data_1['DT'])
fig = plt.figure()
sns.distplot(data_1['ST'])
fig = plt.figure()

model_4 = smf.ols("ST~DT",data = data_3).fit()

model_4.summary()
