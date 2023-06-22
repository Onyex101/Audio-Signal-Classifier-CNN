import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_pills import pills
from PIL import Image
import numpy as np
import librosa
import os
import tensorflow as tf


icon_image = Image.open("streamlit_app/audiosignal.png")
header_image = Image.open("streamlit_app/header_img.png")

st.set_page_config(
    page_title="Audio Signal Classifier",
    layout="wide",
    initial_sidebar_state="auto",
)

result_classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/audio_model.h5')
    
model = load_model()

def save_file(sound_file):
    # save your sound file in the right folder by following the path
    with open("streamlit_app/testfile.wav",'wb') as f:
         f.write(sound_file.getbuffer())
    return "testfile.wav"

with st.sidebar:
    st.header("Audio Signal Classifier")
    st.image(icon_image)
    st.divider()
    uploaded_file = st.file_uploader("Choose File", type=["wav"],accept_multiple_files=False)
    sample_file = st.checkbox('Load Sample file')
    

st.image(header_image)
st.title("Audio Sample Analysis")
classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
pills("Available Classes", classes, index=None)
if sample_file or uploaded_file is not None:
    audio_bytes = None
    if sample_file:
        audio_file = open("streamlit_app/dog_bark.wav", "rb")
        audio_bytes = audio_file.read()
        filename = "streamlit_app/dog_bark.wav"
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        filename = f"streamlit_app/{save_file(uploaded_file)}"
    st.audio(audio_bytes)
    with st.spinner('Loading...'):
        original_audio, sample_rate = librosa.load(filename)
        mfccs_features = librosa.feature.mfcc(y=original_audio, sr=sample_rate, n_mfcc=45)
        melspec = librosa.feature.melspectrogram(y=original_audio,sr=sample_rate)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)
        result_array = model.predict(mfccs_scaled_features)
        result = np.argmax(result_array[0])
        prediction = result_classes[result]
        col_1, col_2 = st.columns(2)
        with col_1:
            st.success(f"Predicted Class: {prediction}", icon="✅")

        with col_2:
            pass
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(nrows=1,figsize=(10, 4),sharex=True)
            librosa.display.waveshow(original_audio,sr=sample_rate,ax=ax)
            ax.set(title='Waveform')
            ax.set_ylabel('amplitude')
            ax.set_xlabel('Time [secs]')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(nrows=1,figsize=(10, 4),sharex=True)
            S_dB = librosa.power_to_db(melspec, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time',
                                y_axis='mel', sr=sample_rate,
                                fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            st.pyplot(fig)
            
        col3, col4 = st.columns(2)

        with col3:
            fig2, ax = plt.subplots(nrows=1,figsize=(10, 4),sharex=True)
            D = librosa.amplitude_to_db(melspec, ref=np.max)
            img = librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sample_rate, ax=ax)
            ax.set(title='Linear-frequency power spectrogram')
            st.pyplot(fig2)
else:
    st.text("Please load audio file in .wav format")
