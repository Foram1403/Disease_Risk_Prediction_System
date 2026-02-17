# ==========================================
# Train Heart Disease Model
# ==========================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

data = pd.read_csv("heart_cleveland_upload.csv")

# Rename target column (important!)
data = data.rename(columns={"condition": "target"})

# ------------------------------------------
# 2. Separate Features and Target
# ------------------------------------------

X = data.drop("target", axis=1)
y = data["target"]

# ------------------------------------------
# 3. Handle Class Imbalance (SMOTE)
# ------------------------------------------

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ------------------------------------------
# 4. Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.25,
    random_state=42
)

# ------------------------------------------
# 5. Scaling
# ------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------
# 6. Gradient Boosting Model
# ------------------------------------------

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ------------------------------------------
# 7. Accuracy
# ------------------------------------------

accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print("Model Accuracy:", accuracy)

# ------------------------------------------
# 8. Save Model + Scaler
# ------------------------------------------

pickle.dump(model, open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("heart_scaler.pkl", "wb"))

print("Model and scaler saved successfully!")
