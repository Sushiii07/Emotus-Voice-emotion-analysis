import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import requests

# Replace 'your_audio_file.wav' with the path to your audio file
file_path = '../03-01-03-01-02-01-03.wav'
url = 'http://127.0.0.1:5000/predict'


X, sample_rate = librosa.load(file_path, duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive
livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T
twodim = np.expand_dims(livedf2, axis=2)

response = requests.post(url, json=twodim.tolist())

# Print the response
print(response.json())

# Get the predictions from the response if needed
# predictions = response.json()cl
# print(predictions)
