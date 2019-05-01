
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


dataset = pd.read_csv("googleplaystore.csv")
dataset.drop(labels=['App','Category','Type','Content Rating','Genres','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)
print(dataset.head())


for index, row in dataset.iterrows():
    if not row['Rating'] > 0 or not row['Reviews'].isalnum() :
        dataset.drop(index, axis=0, inplace=True)

for index, row in dataset.iterrows():
    row['Size'].strip('M')
    row['Installs'].rstrip('+')

print(dataset)




X = dataset[['Reviews']].values
print (X)
y = dataset[['Rating']].values.squeeze()
print (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print('the regressor intercept is', regressor.intercept_)
#For retrieving the slope:
print('the regressor coef is' , regressor.coef_)

y_pred = regressor.predict(X)
print(y_pred)

list1 = []
list2 = []
for i in range (1, y_pred.size):
    list1.append(i)
    list2.append(y_pred[i])

list3 = []
list4 = []
for i in range (1, y_train.size):
    list3.append(i)
    list4.append(y_train[i])
#print the result
#df = pd.DataFrame({'Actual': y_train.flatten(), 'Predicted': y_pred.flatten()})
plt.scatter(list1, list2, c='blue')
plt.scatter(list3, list4, c='red')