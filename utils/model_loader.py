import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def load_model(model_name):
    path = os.path.join(BASE_DIR, "models", model_name)
    return pickle.load(open(path, "rb"))
