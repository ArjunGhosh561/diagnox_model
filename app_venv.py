from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)
CORS(app)
l1 = [
    "back pain",
    "constipation",
    "abdominal pain",
    "diarrhoea",
    "mild fever",
    "yellow urine",
    "yellowing of eyes",
    "acute liver failure",
    "fluid overload",
    "swelling of stomach",
    "swelled lymph nodes",
    "malaise",
    "blurred and distorted vision",
    "phlegm",
    "throat irritation",
    "redness of eyes",
    "sinus pressure",
    "runny nose",
    "congestion",
    "chest pain",
    "weakness in limbs",
    "fast heart rate",
    "pain during bowel movements",
    "pain in anal region",
    "bloody stool",
    "irritation in anus",
    "neck pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen legs",
    "swollen blood vessels",
    "puffy face and eyes",
    "enlarged thyroid",
    "brittle nails",
    "swollen extremeties",
    "excessive hunger",
    "extra marital contacts",
    "drying and tingling lips",
    "slurred speech",
    "knee pain",
    "hip joint pain",
    "muscle weakness",
    "stiff neck",
    "swelling joints",
    "movement stiffness",
    "spinning movements",
    "loss of balance",
    "unsteadiness",
    "weakness of one body side",
    "loss of smell",
    "bladder discomfort",
    "foul smell of urine",
    "continuous feel of urine",
    "passage of gases",
    "internal itching",
    "toxic look (typhos)",
    "depression",
    "irritability",
    "muscle pain",
    "altered sensorium",
    "red spots over body",
    "belly pain",
    "abnormal menstruation",
    "dischromic patches",
    "watering from eyes",
    "increased appetite",
    "polyuria",
    "family history",
    "mucoid sputum",
    "rusty sputum",
    "lack of concentration",
    "visual disturbances",
    "receiving blood transfusion",
    "receiving unsterile injections",
    "coma",
    "stomach bleeding",
    "distention of abdomen",
    "history of alcohol consumption",
    "fluid overload",
    "blood in sputum",
    "prominent veins on calf",
    "palpitations",
    "painful walking",
    "pus filled pimples",
    "blackheads",
    "scurring",
    "skin peeling",
    "silver like dusting",
    "small dents in nails",
    "inflammatory nails",
    "blister",
    "red sore around nose",
    "yellow crust ooze",
]

disease = [
    "Fungal infection",
    "Allergy",
    "GERD",
    "Chronic cholestasis",
    "Drug Reaction",
    "Peptic ulcer disease",
    "AIDS",
    "Diabetes",
    "Gastroenteritis",
    "Bronchial Asthma",
    "Hypertension",
    "Migraine",
    "Cervical spondylosis",
    "Paralysis (brain hemorrhage)",
    "Jaundice",
    "Malaria",
    "Chicken pox",
    "Dengue",
    "Typhoid",
    "hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Alcoholic hepatitis",
    "Tuberculosis",
    "Common Cold",
    "Pneumonia",
    "Dimorphic hemorrhoids(piles)",
    "Heart attack",
    "Varicose veins",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Hypoglycemia",
    "Osteoarthritis",
    "Arthritis",
    "(vertigo) Paroxysmal Positional Vertigo",
    "Acne",
    "Urinary tract infection",
    "Psoriasis",
    "Impetigo",
]
diet_dataset = {
    "Fungal infection": "Balanced Diet",
    "Allergy": "Elimination Diet",
    "GERD": ["Low-Acid Diet", "Fiber-rich Foods"],
    "Chronic cholestasis": "Low-Fat Diet",
    "Drug Reaction": "Consult with a healthcare professional",
    "Peptic ulcer disease": "Avoid spicy and acidic foods",
    "AIDS": "Nutrient-dense Diet",
    "Diabetes": "Balanced Diet with controlled carbohydrates",
    "Gastroenteritis": "BRAT Diet (Bananas, Rice, Applesauce, Toast)",
    "Bronchial Asthma": "Anti-inflammatory Diet",
    "Hypertension": "DASH Diet (Dietary Approaches to Stop Hypertension)",
    " Migraine": "Migraine Diet (Avoiding trigger foods)",
    "Cervical spondylosis": "Anti-inflammatory Diet",
    "Paralysis (brain hemorrhage)": "Balanced Diet with emphasis on antioxidants",
    "Jaundice": "Low-Fat and Low-Protein Diet",
    "Malaria": "High-Protein Diet",
    "Chicken pox": "Soft and Easy-to-Swallow Foods",
    "Dengue": "Fluid and Nutrient-Rich Diet",
    "Typhoid": "Bland and Soft Diet",
    "hepatitis A": "Low-Fat Diet",
    "Hepatitis B": "Low-Fat Diet",
    "Hepatitis C": "Low-Fat Diet",
    "Hepatitis D": "Low-Fat Diet",
    "Hepatitis E": "Low-Fat Diet",
    "Alcoholic hepatitis": "Abstain from alcohol, Low-Fat Diet",
    "Tuberculosis": "High-Calorie and High-Protein Diet",
    "Common Cold": "Adequate Fluids, Vitamin C-rich Foods",
    "Pneumonia": "Balanced Diet with Protein",
    "Dimorphic hemorrhoids (piles)": "High-Fiber Diet",
    "Heart attack": "Heart-Healthy Diet (Low-Sodium, Low-Fat)",
    "Varicose veins": "High-Fiber Diet",
    "Hypothyroidism": "Iodine-rich Diet",
    "Hyperthyroidism": "Iodine-restricted Diet",
    "Hypoglycemia": "Frequent, Balanced Meals",
    "Osteoarthritis": "Anti-inflammatory Diet",
    "Arthritis": "Anti-inflammatory Diet",
    "(Vertigo) Paroxysmal Positional Vertigo": "Low-Salt Diet",
    "Acne": "Low-Glycemic Diet",
    "Urinary tract infection": "Adequate Fluids, Cranberry Juice",
    "Psoriasis": "Anti-inflammatory Diet",
    "Impetigo": "Balanced Diet with emphasis on Vitamins A and C",
}
doctors = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist/Immunologist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist",
    "Drug Reaction": "Allergist/Immunologist",
    "Peptic ulcer diseae": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    " Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedic Surgeon",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Hepatologist",
    "Malaria": "Infectious Disease Specialist",
    "Chicken pox": "Infectious Disease Specialist",
    "Dengue": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "hepatitis A": "Hepatologist",
    "Hepatitis B": "Hepatologist",
    "Hepatitis C": "Hepatologist",
    "Hepatitis D": "Hepatologist",
    "Hepatitis E": "Hepatologist",
    "Alcoholic hepatitis": "Hepatologist",
    "Tuberculosis": "Pulmonologist",
    "Common Cold": "Internal Medicine Specialist",
    "Pneumonia": "Pulmonologist",
    "Dimorphic hemmorhoids(piles)": "Proctologist",
    "Heartattack": "Cardiologist",
    "Varicoseveins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthristis": "Rheumatologist",
    "Arthritis": "(vertigo) Paroymsal  Positional Vertigo",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist",
}
# Load training data
df = pd.read_csv("Training.csv")
df.replace(
    {
        "prognosis": {
            "Fungal infection": 0,
            "Allergy": 1,
            "GERD": 2,
            "Chronic cholestasis": 3,
            "Drug Reaction": 4,
            "Peptic ulcer diseae": 5,
            "AIDS": 6,
            "Diabetes ": 7,
            "Gastroenteritis": 8,
            "Bronchial Asthma": 9,
            "Hypertension ": 10,
            "Migraine": 11,
            "Cervical spondylosis": 12,
            "Paralysis (brain hemorrhage)": 13,
            "Jaundice": 14,
            "Malaria": 15,
            "Chicken pox": 16,
            "Dengue": 17,
            "Typhoid": 18,
            "hepatitis A": 19,
            "Hepatitis B": 20,
            "Hepatitis C": 21,
            "Hepatitis D": 22,
            "Hepatitis E": 23,
            "Alcoholic hepatitis": 24,
            "Tuberculosis": 25,
            "Common Cold": 26,
            "Pneumonia": 27,
            "Dimorphic hemmorhoids(piles)": 28,
            "Heart attack": 29,
            "Varicose veins": 30,
            "Hypothyroidism": 31,
            "Hyperthyroidism": 32,
            "Hypoglycemia": 33,
            "Osteoarthristis": 34,
            "Arthritis": 35,
            "(vertigo) Paroymsal  Positional Vertigo": 36,
            "Acne": 37,
            "Urinary tract infection": 38,
            "Psoriasis": 39,
            "Impetigo": 40,
        }
    },
    inplace=True,
)
X = df[l1]
y = df[["prognosis"]]
np.ravel(y)

# Load testing data
tr = pd.read_csv("Testing.csv")
tr.replace(
    {
        "prognosis": {
            "Fungal infection": 0,
            "Allergy": 1,
            "GERD": 2,
            "Chronic cholestasis": 3,
            "Drug Reaction": 4,
            "Peptic ulcer diseae": 5,
            "AIDS": 6,
            "Diabetes ": 7,
            "Gastroenteritis": 8,
            "Bronchial Asthma": 9,
            "Hypertension ": 10,
            "Migraine": 11,
            "Cervical spondylosis": 12,
            "Paralysis (brain hemorrhage)": 13,
            "Jaundice": 14,
            "Malaria": 15,
            "Chicken pox": 16,
            "Dengue": 17,
            "Typhoid": 18,
            "hepatitis A": 19,
            "Hepatitis B": 20,
            "Hepatitis C": 21,
            "Hepatitis D": 22,
            "Hepatitis E": 23,
            "Alcoholic hepatitis": 24,
            "Tuberculosis": 25,
            "Common Cold": 26,
            "Pneumonia": 27,
            "Dimorphic hemmorhoids(piles)": 28,
            "Heart attack": 29,
            "Varicose veins": 30,
            "Hypothyroidism": 31,
            "Hyperthyroidism": 32,
            "Hypoglycemia": 33,
            "Osteoarthristis": 34,
            "Arthritis": 35,
            "(vertigo) Paroymsal  Positional Vertigo": 36,
            "Acne": 37,
            "Urinary tract infection": 38,
            "Psoriasis": 39,
            "Impetigo": 40,
        }
    },
    inplace=True,
)
X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)


# Load trained models
def load_model(filename):
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model


# Specify monotonic constraints for features
monotonic_constraints = [
    0,
    1,
    -1,
    1,
    0,
    0,
    ...,
]  # Replace ... with the constraints for other features
clf3 = load_model("decision_tree_model.pkl")
clf4 = load_model("random_forest_model.pkl")
gnb = load_model("naive_bayes_model.pkl")


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
@app.route("/")
def afd():
    return "moye moye"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.json["symptoms"]
        symptoms = [s.strip() for s in user_input.split(",")]

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
            "decisionTree": decision_tree_accuracy,
            "randomForest": random_forest_accuracy,
            "naiveBayes": naive_bayes_accuracy,
        }
        nb_models_accuracies = {
            "naiveBayesPred": naive_bayes_prediction,
            "accuracy": naive_bayes_accuracy,
            "dietsPrescribed": diet_dataset[naive_bayes_prediction],
            "doctor": doctors[naive_bayes_prediction],
            
        }

        dt_models_accuracies = {
            "decisionTreePred": decision_tree_prediction,
            "accuracy": decision_tree_accuracy,
            "dietsPrescribed": diet_dataset[decision_tree_prediction],
            "doctor": doctors[decision_tree_prediction],
        }
        rf_models_accuracies = {
            "randomForestPred": random_forest_prediction,
            "accuracy": random_forest_accuracy,
            "dietsPrescribed": diet_dataset[naive_bayes_prediction],
            "doctor": doctors[random_forest_prediction],
        }

        best_model = max(models_accuracies, key=models_accuracies.get)
        best_accuracy = models_accuracies[best_model]
        best_prediction = locals()[f"{best_model.lower().replace(' ', '_')}_prediction"]

        result = {
            "bestModel": best_model,
            "bestAccuracy": best_accuracy,
            "bestPrediction": best_prediction,
        }

        return jsonify(
            nb_models_accuracies, dt_models_accuracies, rf_models_accuracies, result
        )

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
