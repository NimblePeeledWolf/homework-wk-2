from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
model = pickle.load(open('svm_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form and convert them to a list
    input_features = [float(x) for x in request.form.values()]
    # Predict the outcome using the SVM model
    prediction = model.predict([input_features])
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
