import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

x_train = np.array([1.0,2.0,3.0,2.5,4.0,5.0,6.5,7.5,8.0,9.0,9.5,10.0,2.5,2.0])
y_train = np.array([300.0,400.0,700.0,750.0,550.0,625.0,780.0,870,910,790,890,1020,300.0,240.0])
print('X TRAIN',x_train)
print('Y TRAIN',y_train)
print('Shape is ',x_train.shape)
m = x_train.shape[0]
print('Training Examples',m)
i = 0
x_i = x_train[i]
y_i = y_train[i]
print('X_I',x_i,'y_i',y_i)

w = 85
b = 215
print('W',w,'B',b)

def compute_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

tmp_fwb = compute_output(x_train,w,b)

print('House price predictor')
x_i = int(input('Enter size of house in square feet : '))/1000

cost = w * x_i + b
print(f'Predicted price of your house is : {cost * 100} dollars')
plt.plot(x_train,tmp_fwb,c='b',label='My predictions')

plt.scatter(x_train,y_train,marker='x',c='g', label='Actual values')

plt.title('House prices')

plt.ylabel('Price in 1000s dollars')

plt.xlabel('House sizes in 1000s sqft')
plt.legend()
#plt.show()