# Step 1: Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 2: Generate a synthetic dataset
#X1 HOUSE SIZE
#X2 Num of bedrooms
#y Price
#np.random.seed(42)
X1 = np.array([2150,2300,1980,2040,2500,1300,2011,1860,3000,2640])
X2 = np.array([4,5,3,4,5,2,3,4,6,6])
y = np.array([153200, 185500, 137800, 162300, 145700, 175900, 149600, 141400, 180200, 155000])
#print('y',y.shape)
# Step 3: No need to create a DataFrame; work directly with NumPy arrays
# Step 4: Split the dataset into training and testing sets
X = np.column_stack([X1, X2])  # Features (independent variables)
#y = y.flatten()          # Target (dependent variable)
#print('X',X.shape)
#print('y',y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Standardize the features (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#print('X1',X_train_scaled.shape)
#print('X',X_test_scaled.shape)
# # Step 6: Create and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# # Step 7: Make predictions on the test set
y_pred = model.predict(X_test_scaled)
#print(y_pred.shape)
# # Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#print(f'Mean Squared Error: {mse}')
#print(f'R-squared: {r2}')

house_size = int(input('Enter house size :'))
no_bedrooms = int(input('Enter no of bedrooms in the house :'))
#Scaling
new_data = scaler.transform([[house_size,no_bedrooms]])
price_predicted = model.predict(new_data)
print('Price predicted is : ',int(price_predicted),'Dollars')