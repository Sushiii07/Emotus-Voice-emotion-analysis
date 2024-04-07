import streamlit as st
import requests
import numpy as np
from io import BytesIO
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import streamlit.components.v1 as components
import base64
from streamlit_mic_recorder import mic_recorder
from streamlit_mic_recorder import speech_to_text

FLASK_URL = 'http://127.0.0.1:5000/predict'
label_encoder = joblib.load('../label_encoder.pkl')

def feature_graph(data):
    librosa.display.waveshow(data, sr=sampling_rate, color="blue")

def preprocess_audio(audio_file, glb):
    X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    glb['twodim'] = twodim
    glb['X'] = X

def callback():
    if st.session_state.my_recorder_output:
        audio_bytes = st.session_state.my_recorder_output['bytes']
        st.audio(audio_bytes)


def main():
    st.set_page_config(layout="wide")
    st.title('Audio Emotion Recognition')


    # Custom CSS to decrease padding
    padding_css = """
    <style>
    .main  {
        margin-left: 10rem;
        margin-right: 10rem;
    }
    </style>
    """
    st.markdown(padding_css, unsafe_allow_html=True)

    # Define the layout with wider columns
    # col1, _, col2, _ = st.columns([2, 0.2, 2, 0.2])
    col1, col2 = st.columns([2, 2])

    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=False,
        format="wav",
        callback=None,
        args=(),
        kwargs={},
        key=None
    )

    text = speech_to_text(
        language='en',
        start_prompt="Start speech-to-text",
        stop_prompt="Stop speech-to-text",
        just_once=False,
        use_container_width=False,
        callback=None,
        args=(),
        kwargs={},
        key=None
    )

    # Input column
    glb = {}
    with col1:
        st.header('Input')
        uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

        if uploaded_file is not None:
            st.audio(uploaded_file)

            if st.button('Predict'):
                audio_bytes = uploaded_file.read()
                # glb = {}
                preprocess_audio(BytesIO(audio_bytes), glb)
                if 'twodim' in glb:
                    response = requests.post(FLASK_URL, json=glb['twodim'].tolist())
                    emotions = response.json()
                    # st.write(emotions)
                    global_labels = label_encoder.inverse_transform(np.array(range(0, 10)))
                    st.table(pd.DataFrame({"Emotions": global_labels, "Probable Values": emotions[0]}))
        
        elif audio is not None:
            if st.button('Predict'):
                st.text(text)
                audio_bytes = audio["bytes"]
                preprocess_audio(BytesIO(audio_bytes), glb)
                if 'twodim' in glb:
                    response = requests.post(FLASK_URL, json=glb['twodim'].tolist())
                    emotions = response.json()
                    # st.write(emotions)
                    global_labels = label_encoder.inverse_transform(np.array(range(0, 10)))
                    _emots = {}
                    for i, v in enumerate(emotions[0]):
                        _emots[global_labels[i]] = v
                    # st.write(_emots)
                    st.table(pd.DataFrame({"Emotions": global_labels, "Probable Values": emotions[0]}))

            # mic_recorder(key='my_recorder', callback=callback)

    # Output column
    with col2:
        st.header('Output')
        if 'twodim' in glb:
            emotions = np.array(emotions)
            pred_labels = label_encoder.inverse_transform(np.argmax(emotions, axis=1))
            pred_labels = pred_labels[0].replace('_', ' ')
            st.success("Verdict: " + pred_labels.capitalize())

            # plt.figure(figsize=(15, 5))
            fig2, ax = plt.subplots(figsize=(10, 6))
            librosa.display.waveshow(glb['X'], sr=22050, color="blue")  
            st.pyplot(fig2)
            fig, ax = plt.subplots(figsize=(10, 6))  
            ax.bar(label_encoder.classes_, emotions.flatten())
            ax.set_xlabel('Categories')
            ax.set_xticklabels(label_encoder.classes_, rotation=90)
            ax.set_ylabel('Values')
            st.pyplot(fig)

if __name__ == '__main__':
    main()
