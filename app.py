from flask import Flask, request, render_template
from joblib import load
import numpy as np
import pickle

app = Flask(__name__)

# Carico il modello e lo scaler
with open('rfr.pkl', 'rb') as f:
    rfr = pickle.load(f)

scaler = load('scaler.joblib')

cut_mapping = {
    'Fair': [1, 0, 0, 0, 0],
    'Good': [0, 1, 0, 0, 0],
    'Ideal': [0, 0, 1, 0, 0],
    'Premium': [0, 0, 0, 1, 0],
    'Very Good': [0, 0, 0, 0, 1]
}

color_mapping = {
    'D': [1, 0, 0, 0, 0, 0, 0],
    'E': [0, 1, 0, 0, 0, 0, 0],
    'F': [0, 0, 1, 0, 0, 0, 0],
    'G': [0, 0, 0, 1, 0, 0, 0],
    'H': [0, 0, 0, 0, 1, 0, 0],
    'I': [0, 0, 0, 0, 0, 1, 0],
    'J': [0, 0, 0, 0, 0, 0, 1]
}

clarity_mapping = {
    'I1': [1, 0, 0, 0, 0, 0, 0, 0],
    'IF': [0, 1, 0, 0, 0, 0, 0, 0],
    'SI1': [0, 0, 1, 0, 0, 0, 0, 0],
    'SI2': [0, 0, 0, 1, 0, 0, 0, 0],
    'VS1': [0, 0, 0, 0, 1, 0, 0, 0],
    'VS2': [0, 0, 0, 0, 0, 1, 0, 0],
    'VVS1': [0, 0, 0, 0, 0, 0, 1, 0],
    'VVS2': [0, 0, 0, 0, 0, 0, 0, 1]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Estraggo i dati dal form e scalo le feature numeriche
    carat = float(request.form['carat'])
    table = float(request.form['table'])
    x = float(request.form['x'])
    depth = float(request.form['depth'])
    numeric_input_features = [carat, table, x, depth]
    input_features_scaled = scaler.transform([numeric_input_features])

    # Estraggo e trasformo le feature categoriche
    cut = np.array(cut_mapping[request.form['cut']]).reshape(1, -1)
    color = np.array(color_mapping[request.form['color']]).reshape(1, -1)
    clarity = np.array(clarity_mapping[request.form['clarity']]).reshape(1, -1)


    test = np.concatenate((input_features_scaled, cut, color, clarity), axis=1)

    prediction = rfr.predict(test)

    return render_template('index.html', prediction_text=f'The estimated price is: ${prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
