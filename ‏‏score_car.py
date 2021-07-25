import json
import numpy as np
import os
import onnxruntime
from PIL import Image
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.data_utils import get_file
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication
import pickle as pk


CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

car_list = None


def get_predictions(preds, top=20):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def init():
    global model
    global car_list
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'vgg16_model.h5')
    try:
        model = load_model(model_path)

        svc_pr = ServicePrincipalAuthentication(
            tenant_id="<tenant_id>",
            service_principal_id="<service_principal_id>",
            service_principal_password="<service_principal_password>")

        ws = Workspace.get(name="cardetection",
                           subscription_id='<subscription_id>',
                           resource_group='quickcar',
                           auth=svc_pr)


        datastore = Datastore.get(ws, datastore_name='workspaceblobstore')
        dataset = Dataset.File.from_files((datastore, 'UI/07-09-2021_052755_UTC/car_counter.pk'))
        cars_set = dataset.download()
        with open(cars_set[0], 'rb') as f:
            car_counter = pk.load(f)
        car_list = [k for k, v in car_counter.most_common()[:50]]

    except Exception as e:
        print("Exception:")
        print(str(e))


@rawhttp
def run(request):
    # prepare image
    reqBody = request.get_data(False)
    img = Image.open(BytesIO(reqBody))
    img = img.resize((224, 224))

    # adjust to model
    x = np.asarray(img)
    x = np.expand_dims(x, axis=0)

    # predict
    out = model.predict(x)
    top = get_predictions(out, top=5)
    for j in top[0]:
        if j[0:2] in car_list:
            return True, (j[0:2])
    return False