import pickle
import numpy as np

heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
heart_scaler = pickle.load(open("models/heart_scaler.pkl", "rb"))

diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
diabetes_scaler = pickle.load(open("models/diabetes_scaler.pkl", "rb"))

liver_model = pickle.load(open("models/liver_model.pkl", "rb"))
liver_scaler = pickle.load(open("models/liver_scaler.pkl", "rb"))

def predict(disease, features):
    if disease == "Heart":
        model, scaler = heart_model, heart_scaler
    elif disease == "Diabetes":
        model, scaler = diabetes_model, diabetes_scaler
    else:
        model, scaler = liver_model, liver_scaler

    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1] * 100
    return prob
