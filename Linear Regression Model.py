import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd 
#print(plt.style.available)
plt.style.use('seaborn-v0_8-dark')

df = pd.read_csv('Housing.csv')

#x = np.array([74.20,    89.60,  99.60,    75.00,  74.20,   85.80,   162.00,    81.00,  57.50, 132.00  ,60.00, 65.50,  35.00,  78.00])
#y = np.array([13300.00,12250.00,12250.00,12215.00,11410.00,10150.00,10150.000,9870.00,9800.00,9800.00,9681.00,9310.00,9240.00, 9240.00])
#print(f'x_train = {x}')
#print(f'y_train = {y}')
#print(f'x.shape = {x.shape}')
#m = x.shape[0]
#print(f'Traning examples {m}')
col = df['area']
row = df['price']
x = np.array(col)
y = np.array(row)



i = 0
x_i = x[i]
y_i = y[i]
#print(f'x^i is {x_i},y^i is {y_i}')

def computation_model(a,b,x_train):
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    f_wb = a * x[i] + b

    return f_wb
w = 100
b = 100

out_func = computation_model(w,b,x)

x_i = int(input('Enter house size in sqaure feet : ')) / 1000
cost = w * x_i + b
print(cost*100,'dollars')

plt.plot(x,y,marker = 'x',c = 'r', label = 'Actual Values')

plt.title('House predictor')

plt.ylabel('Price in 1000$')

plt.xlabel('Size in sqft')

plt.legend()

plt.show()

