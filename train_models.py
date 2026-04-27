import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

if not os.path.exists("models"):
    os.makedirs("models")

def train(df, target, name):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    pickle.dump(model, open(f"models/{name}.pkl", "wb"))

# LOAD DATA
train(pd.read_csv("data_sets/diabetes dataset.csv"), "Outcome", "diabetes_model")
train(pd.read_csv("data_sets/heart.csv"), "target", "heart_model")
train(pd.read_csv("data_sets/breast_cancer.csv"), "diagnosis", "breast_cancer_model")
train(pd.read_csv("data_sets/lung_cancer.csv"), "LUNG_CANCER", "lung_cancer_model")
