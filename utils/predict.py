import numpy as np

def make_prediction(model, data):
    data = np.array(data).reshape(1, -1)
    return model.predict(data)
