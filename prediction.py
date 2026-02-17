# # importing required libraries
# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier


# # loading and reading the dataset

# heart = pd.read_csv("heart_cleveland_upload.csv")

# # creating a copy of dataset so that will not affect our original dataset.
# heart_df = heart.copy()

# # Renaming some of the columns
# heart_df = heart_df.rename(columns={'condition':'target'})
# print(heart_df.head())

# # model building

# #fixing our data in x and y. Here y contains target data and X contains rest all the features.
# x= heart_df.drop(columns= 'target')
# y= heart_df.target

# # splitting our dataset into training and testing for this we will use train_test_split library.
# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

# #feature scaling
# scaler= StandardScaler()
# x_train_scaler= scaler.fit_transform(x_train)
# x_test_scaler= scaler.fit_transform(x_test)

# # creating K-Nearest-Neighbor classifier
# model=RandomForestClassifier(n_estimators=20)
# model.fit(x_train_scaler, y_train)
# y_pred= model.predict(x_test_scaler)
# p = model.score(x_test_scaler,y_test)
# print(p)

# print('Classification Report\n', classification_report(y_test, y_pred))
# print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# # Creating a pickle file for the classifier
# filename = 'heart-disease-prediction-knn-model.pkl'

# pickle.dump(model, open(filename, 'wb'))
# ==========================================
# AI Multi-Disease Training Script
# Gradient Boosting + SMOTE
# ==========================================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


# ==========================================
# Function to Train Model
# ==========================================

def train_model(dataset_path, target_column, model_name):

    print(f"\nTraining {model_name} model...")

    data = pd.read_csv(dataset_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # -------- Handle Class Imbalance --------
    try:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("SMOTE applied successfully.")
    except:
        print("SMOTE skipped (if already balanced).")

    # -------- Split Data --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # -------- Scaling --------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------- Gradient Boosting Model --------
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # -------- Evaluation --------
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)

    print("Accuracy:", acc)
    print(classification_report(y_test, predictions))

    # -------- Save Model --------
    os.makedirs("models", exist_ok=True)

    pickle.dump(model, open(f"models/{model_name}_model.pkl", "wb"))
    pickle.dump(scaler, open(f"models/{model_name}_scaler.pkl", "wb"))

    print(f"{model_name} model saved successfully.")


# ==========================================
# Train Multiple Diseases
# ==========================================

train_model("datasets/heart.csv", "target", "heart")
train_model("datasets/diabetes.csv", "target", "diabetes")
train_model("datasets/lung.csv", "target", "lung")
