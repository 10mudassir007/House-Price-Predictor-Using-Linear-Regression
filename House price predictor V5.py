#Importing necessary libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression

#Reading DATA
df = pd.read_csv('Housing.csv')

#Applying some feature engineering to make all the features in the same scale 0 to 10
df['sqft'] = df['sqft'] / 1000

x1 = df[['rooms','sqft','floors','condition']]

#Price label y
y = df['price']


#Splitting data into train and test sets by 80,20
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=42)

#Scaling the data
scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#Creating and training the model with scaled data
model = LinearRegression()
model.fit(x_train_scaled,y_train)


#Making predictions
y_hat = model.predict(x_test_scaled)


#Creating function for R2 score for training data
def adj_R2(x_train_scaled,y_train):
  r2 = model.score(x_train_scaled,y_train)
  n = x_train_scaled.shape[0]
  p = x_train_scaled.shape[1]
  adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
  return adjusted_r2


#Creating function for R2 score for test data
def adj_R2_test(x_test_scaled,y_test):
  r2 = model.score(x_test_scaled,y_test)
  n = x_test_scaled.shape[0]
  p = x_test_scaled.shape[1]
  adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
  return adjusted_r2


adj_R2 = adj_R2(x_train_scaled,y_train)

adj_R2_test = adj_R2_test(x_test_scaled,y_test)

#Visualizing data (if necessary)
import matplotlib.pyplot as plt

#Visualizing differences between predictions and actual
plt.hist(y_hat - y_test)
plt.xlabel("Difference between Actual and Predicted Price")
plt.ylabel("Frequency")
#plt.show()

#Some error metrics to explore
# print('R2-Train                      : {0:.2f}'.format(model.score(x_train_scaled, y_train)*100))
# print('R2-Test                       : {0:.2f}'.format(model.score(x_test_scaled, y_test)*100))
# print('Adj_R2-Train                  : {0:.2f}'.format(adj_R2*100))
# print('Adj_R2-Test                   : {0:.2f}'.format(adj_R2_test*100))
# print('MSE (Mean Squared Error)      : {0:.0f}'.format(metrics.mean_squared_error(y_test, y_hat)))
# print('RMSE (Root Mean Squared Error): {0:.0f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_hat))))
# print('MAE (Mean Ablosute Error)     : {0:.0f}'.format(metrics.mean_absolute_error(y_test, y_hat)))


#Input features
house_size = int(input('Enter size of house in square feet : ')) / 1000
rooms = int(input('Enter rooms in house : '))
floors = int(input('Enter floors in house : '))
condition = int(input('Enter condition of a house out of 10 : '))
features = [[rooms,house_size,floors,condition]]

#Scaling Features
scaled_features = scaler.transform(features)

#Using model to predict the new data
price = model.predict(scaled_features)
print(f'Price of the house is : {int(price)} dollars')

