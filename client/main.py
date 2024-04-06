import streamlit as st
import requests
import numpy as np
from io import BytesIO
import librosa
import pandas as pd

# Define the URL of your Flask backend
FLASK_URL = 'http://127.0.0.1:5000/predict'

def preprocess_audio(audio_file, glb):
    # Load audio file using librosa
    X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    # # Extract MFCC features using librosa
    # mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13)
    # # Calculate mean MFCCs
    # mean_mfccs = np.mean(mfccs, axis=1)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    glb['twodim'] = twodim
    st.text(twodim)

def predict(features):
    # Assume you have a function to make predictions using the features
    # predictions = model.predict(features)
    predictions = ['happy', 'sad', 'angry']  # Example predictions
    return predictions

def main():
    st.title('(^_^)')

    glb = {}

    # Upload audio file
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Display uploaded audio file
        st.audio(uploaded_file)

        # Perform prediction when button is clicked
        if st.button('Predict'):
            # Preprocess uploaded file
            audio_bytes = uploaded_file.read()
            preprocess_audio(BytesIO(audio_bytes), glb)
            # Make predictions
            if 'twodim' in glb:
                response = requests.post(FLASK_URL, json=glb['twodim'].tolist())
                # predictions = predict(glb["twodim"])
                # Display predictions
                st.write(response.json())
            else:
                st.error("Prank Ho Gaya BC")

if __name__ == '__main__':
    main()
