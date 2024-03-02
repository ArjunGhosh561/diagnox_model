from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

l1 = ['back pain', 'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine',
      'yellowing of eyes', 'acute liver failure', 'fluid overload', 'swelling of stomach',
      'swelled lymph nodes', 'malaise', 'blurred and distorted vision', 'phlegm', 'throat irritation',
      'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'chest pain', 'weakness in limbs',
      'fast heart rate', 'pain during bowel movements', 'pain in anal region', 'bloody stool',
      'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen legs',
      'swollen blood vessels', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails',
      'swollen extremeties', 'excessive hunger', 'extra marital contacts', 'drying and tingling lips',
      'slurred speech', 'knee pain', 'hip joint pain', 'muscle weakness', 'stiff neck', 'swelling joints',
      'movement stiffness', 'spinning movements', 'loss of balance', 'unsteadiness',
      'weakness of one body side', 'loss of smell', 'bladder discomfort', 'foul smell of urine',
      'continuous feel of urine', 'passage of gases', 'internal itching', 'toxic look (typhos)',
      'depression', 'irritability', 'muscle pain', 'altered sensorium', 'red spots over body', 'belly pain',
      'abnormal menstruation', 'dischromic patches', 'watering from eyes', 'increased appetite', 'polyuria',
      'family history', 'mucoid sputum',
      'rusty sputum', 'lack of concentration', 'visual disturbances', 'receiving blood transfusion',
      'receiving unsterile injections', 'coma', 'stomach bleeding', 'distention of abdomen',
      'history of alcohol consumption', 'fluid overload', 'blood in sputum', 'prominent veins on calf',
      'palpitations', 'painful walking', 'pus filled pimples', 'blackheads', 'scurring', 'skin peeling',
      'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose',
      'yellow crust ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria',
           'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
           'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
           'Dimorphic hemorrhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism',
           'Hypoglycemia', 'Osteoarthritis', 'Arthritis', '(vertigo) Paroxysmal Positional Vertigo', 'Acne',
           'Urinary tract infection', 'Psoriasis', 'Impetigo']

# Load training data
df = pd.read_csv('Training.csv')
df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3,
                          'Drug Reaction': 4, 'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7,
                          'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
                          'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18,
                          'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22,
                          'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26,
                          'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
                          'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36,
                          'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40}},
           inplace=True)
X = df[l1]
y = df[["prognosis"]]
np.ravel(y)

# Load testing data
tr = pd.read_csv('Testing.csv')
tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3,
                          'Drug Reaction': 4, 'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7,
                          'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
                          'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18,
                          'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22,
                          'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26,
                          'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
                          'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36,
                          'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40}},
           inplace=True)
X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

# Load trained models
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

clf3 = load_model('decision_tree_model.pkl')
clf4 = load_model('random_forest_model.pkl')
gnb = load_model('naive_bayes_model.pkl')

def predict_disease(symptoms, model):
    l2 = [0] * len(l1)

    for symptom in symptoms:
        if symptom in l1:
            index = l1.index(symptom)
            l2[index] = 1

    input_test = [l2]
    prediction = model.predict(input_test)
    predicted = prediction[0]

    return disease[predicted]

def calculate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Define the route for receiving symptoms via POST request
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json['symptoms']
        symptoms = [s.strip() for s in user_input.split(',')]

        # Decision Tree
        decision_tree_prediction = predict_disease(symptoms, clf3)
        decision_tree_accuracy = calculate_accuracy(clf3, X_test, np.ravel(y_test))

        # Random Forest
        random_forest_prediction = predict_disease(symptoms, clf4)
        random_forest_accuracy = calculate_accuracy(clf4, X_test, np.ravel(y_test))

        # Naive Bayes
        naive_bayes_prediction = predict_disease(symptoms, gnb)
        naive_bayes_accuracy = calculate_accuracy(gnb, X_test, np.ravel(y_test))

        # Choose the model with the highest accuracy
        models_accuracies = {
            "Decision Tree": decision_tree_accuracy,
            "Random Forest": random_forest_accuracy,
            "Naive Bayes": naive_bayes_accuracy
        }
        nb_models_accuracies ={
            "Naive Bayes Pred": naive_bayes_prediction,
            "Accuracy":naive_bayes_accuracy
        }

        dt_models_accuracies = {
            "Decision Tree Pred":decision_tree_prediction,
            "Accuracy":decision_tree_accuracy
        }
        rf_models_accuracies ={
            "Random Forest Pred": random_forest_prediction,
            "Accuracy":random_forest_accuracy
        }
        

        best_model = max(models_accuracies, key=models_accuracies.get)
        best_accuracy = models_accuracies[best_model]
        best_prediction = locals()[f"{best_model.lower().replace(' ', '_')}_prediction"]
        
        result = {
            "Best Model": best_model,
            "Best Accuracy": best_accuracy,
            "Best Prediction": best_prediction
        }
        

        return jsonify(nb_models_accuracies,dt_models_accuracies,rf_models_accuracies,result) 

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
