import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
import joblib

model = tf.keras.models.load_model("saved_models/Emotion_Voice_Detection_Model.h5")
lb = joblib.load('label_encoder.pkl')

X, sample_rate = librosa.load('03-01-03-01-02-01-03.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive
livedf2= pd.DataFrame(data=livedf2)
livedf2 = livedf2.stack().to_frame().T
twodim= np.expand_dims(livedf2, axis=2)
livepreds = model.predict(twodim, batch_size=32, verbose=1)
print(livepreds)

preds1=livepreds.argmax(axis=1)
print(preds1)
abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))
print(predictions)

# preds = loaded_model.predict(x_testcnn, batch_size=32, 
#                          verbose=1)
