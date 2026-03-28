import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Create models directory
os.makedirs("models", exist_ok=True)

# -------------------------------
# COMMON TRAIN FUNCTION
# -------------------------------
def train_and_save(df, target_col, model_name):
    print(f"\n🔹 Training {model_name} model...")

    # Split features & target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ {model_name} Accuracy: {acc:.2f}")

    # Save model
    with open(f"models/{model_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler
    with open(f"models/{model_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"📁 Saved: {model_name}_model.pkl & {model_name}_scaler.pkl")


# -------------------------------
# 1️⃣ HEART DISEASE
# -------------------------------
heart_df = pd.read_csv("data/heart.csv")

# Ensure target column
if "target" not in heart_df.columns:
    raise Exception("Heart dataset must contain 'target' column")

train_and_save(heart_df, "target", "heart")


# -------------------------------
# 2️⃣ DIABETES
# -------------------------------
diabetes_df = pd.read_csv("data/diabetes.csv")

# Rename target column
if "Outcome" in diabetes_df.columns:
    diabetes_df.rename(columns={"Outcome": "target"}, inplace=True)
elif "target" not in diabetes_df.columns:
    raise Exception("Diabetes dataset must contain 'Outcome' or 'target'")

train_and_save(diabetes_df, "target", "diabetes")


# -------------------------------
# 3️⃣ LIVER DISEASE
# -------------------------------
liver_df = pd.read_csv("data/liver.csv")

# Rename target
if "Dataset" in liver_df.columns:
    liver_df.rename(columns={"Dataset": "target"}, inplace=True)

# Convert to binary (1 = disease, 0 = no disease)
liver_df["target"] = liver_df["target"].apply(lambda x: 1 if x == 1 else 0)

train_and_save(liver_df, "target", "liver")


print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
