from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


scaler = StandardScaler()  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form data
    features = np.array([
        float(request.form['Clump_thickness']),
        float(request.form['Uniformity_of_cell_size']),
        float(request.form['Uniformity_of_cell_shape']),
        float(request.form['Marginal_adhesion']),
        float(request.form['Single_epithelial_cell_size']),
        float(request.form['Bare_nuclei']),
        float(request.form['Bland_chromatin']),
        float(request.form['Normal_nucleoli']),
        float(request.form['Mitoses'])
    ]).reshape(1, -1)

    # Standardize features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    # Return result
    result = "Malignant" if prediction[0] == 4 else "Benign"
    return render_template('index.html', result=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
