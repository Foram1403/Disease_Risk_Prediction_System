import pickle
import os

def load_model(model_name):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(BASE_DIR, "models", model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return pickle.load(open(model_path, "rb"))
