import streamlit as st
import requests
import numpy as np
from io import BytesIO
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import joblib

FLASK_URL = 'http://127.0.0.1:5000/predict'
label_encoder = joblib.load('../label_encoder.pkl')

def preprocess_audio(audio_file, glb):
    X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    # mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13)
    # mean_mfccs = np.mean(mfccs, axis=1)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    glb['twodim'] = twodim
    # st.text(twodim)


# def predict(features):
#     # predictions = model.predict(features)
#     predictions = ['happy', 'sad', 'angry']  # Example predictions
#     return predictions

def main():
    st.title('Audio Emotion Recognition')

    glb = {}

    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button('Predict'):
            audio_bytes = uploaded_file.read()
            preprocess_audio(BytesIO(audio_bytes), glb)
            # Make predictions
            if 'twodim' in glb:
                response = requests.post(FLASK_URL, json=glb['twodim'].tolist())
                # predictions = predict(glb["twodim"])
                emotions = response.json()
                st.write(emotions)
                emotions = np.array(emotions)
                preds1 = emotions.argmax(axis=1)
                # print(preds1)
                abc = preds1.astype(int).flatten()
                predictions = (label_encoder.inverse_transform(([abc])))
                graph_labels = (label_encoder.inverse_transform(([0,1,2,3,4,5,6,7,8,9])))
                # print(predictions)
                st.text("Verdict: " + predictions[0].replace('_', ' '))
                fig, ax = plt.subplots()
                ax.bar(graph_labels, emotions.flatten())
                ax.set_xlabel('Categories')
                ax.set_xticklabels(graph_labels, rotation=90)
                ax.set_ylabel('Values')
                st.pyplot(fig)
                # plt.bar(x=emotions.flatten(), height=10)
                # st.text(emotions.flatten())
                # st.pyplot(plt)
                chart_data = pd.DataFrame(emotions.flatten(), columns=["a"])
                # st.area_chart(chart_data)
                # st.map(emotions.flatten())

                # st.bar_chart(x=graph_labels, y=emotions.flatten())

            else:
                st.error("Some error occured")


if __name__ == '__main__':
    main()
