import json
import numpy as np
import os
from PIL import Image
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import img_to_array

def init():
    global model
    weights_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'damage_detector_weights.h5')
    try:
        model = Sequential([
            Convolution2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Convolution2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Convolution2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(1, activation='sigmoid')

        ])
        model.load_weights(weights_path)
    except Exception as e:
        print("Exception")
        print(str(e))


@rawhttp
def run(request):
    try:
        # prepare image
        reqBody = request.get_data(False)
        img = Image.open(BytesIO(reqBody))
        img = img.resize((150, 150))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape) / 255

        pred = model.predict(x)
        print ("prediction: " + str(pred))
        if pred[0][0] <= .5:
            return True
        else:
            return False
    except Exception as e:
        print("Exception:")
        print(e)
        return None
