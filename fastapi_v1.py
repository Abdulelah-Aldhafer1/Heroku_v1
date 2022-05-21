import uvicorn
import tensorflow

from keras.models import load_model
import joblib
import tensorflow
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import librosa
from fastapi import FastAPI, File, UploadFile, Form
import shutil
import librosa

import nest_asyncio


def feature_extraAlldata(aud, Method):
    signal, sr = librosa.load(aud, sr=22050, duration=3, res_type='kaiser_fast')

    # MFCC
    if Method == "mfcc":

        features = librosa.feature.mfcc(signal, sr=sr, n_mfcc=128, hop_length=512).T
    else:
        ## MelSpectogram
        features = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128, fmax=8000, hop_length=512).T

    melsp_matrix = np.array(features)  # chose  MFCC or Mel
    return melsp_matrix


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


import nest_asyncio

nest_asyncio.apply()  # 2. Create the app object
app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post('/predict')
async def predict_banknote(file: UploadFile = File(...)):
    with open(f'{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    feautre1 = feature_extraAlldata(file.filename, "mfcc")
    feautre1 = np.array(feautre1)
    classifier = load_model('CnnRnn.h5')
    features1 = pd.DataFrame(feautre1)  # make it pretty for display
    feautre1 = np.reshape(feautre1, (1, 130, 128))
    pre1 = classifier.predict(feautre1)
    array = largest_indices(pre1, 3)
    with open('dict.pickle', 'rb') as handle:
        inv_map = pickle.load(handle)
    predictions = []
    predictions.append(array[1][0])
    predictions.append(array[1][1])
    predictions.append(array[1][2])
    A = [str(x) for x in predictions]
    if np.max(pre1) > 0.8:
        return {
            'prediction': A
        }
    else:
        return {
            'prediction': "none"
        }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)