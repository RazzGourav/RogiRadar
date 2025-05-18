import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model, label encoder, and symptom vocabulary
model = joblib.load('./model.pkl')
label_encoder = joblib.load('./label_encoder.pkl')
symptom_list = joblib.load('./symptom_vocab.pkl')  # Assuming you saved symptom list here

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get selected symptoms from the form
        symptoms = [request.form.get(f'symptom{i}') for i in range(1, 18)]
        # Clean: lowercase, strip, and remove empty
        symptoms = [s.lower().strip() for s in symptoms if s and s.strip() != '']

        # Create a binary input vector matching symptom_list order and length
        input_vector = [0] * len(symptom_list)
        for symptom in symptoms:
            if symptom in symptom_list:
                idx = symptom_list.index(symptom)
                input_vector[idx] = 1
            else:
                # You may want to handle unknown symptoms here
                pass

        # Predict encoded label
        pred_label = model.predict([input_vector])[0]

        # Decode label to disease name
        prediction = label_encoder.inverse_transform([pred_label])[0]

    # Render the template with symptom_list and prediction if any
    return render_template('index.html', symptom_list=symptom_list, prediction=prediction)

