import numpy as np

def predict(model, inputs):
    data = np.array(inputs).reshape(1, -1)
    return model.predict(data)
