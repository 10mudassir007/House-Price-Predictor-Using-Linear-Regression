import numpy as np
import pickle
from flask import Flask,request,render_template

app = Flask(__name__)

with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl','rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    house_size = float(request.form['house_size']) / 1000
    rooms = int(request.form['rooms'])
    floors = int(request.form['floors'])
    condition = int(request.form['condition'])

    features = np.array([[rooms,house_size,floors,condition]])
    scaled_features = scaler.transform(features)

    price = model.predict(scaled_features)

    return render_template('index.html',prediction=f'Price of the house is : {int(price)} dollars')

if __name__ == '__main__':
    app.run(debug=True,port=5001)