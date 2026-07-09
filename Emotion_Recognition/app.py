import streamlit as st
import numpy as np
import librosa
import joblib

model = joblib.load("emotion_model.pkl")
encoder = joblib.load("label_encoder.pkl")

def extract_features(file):

    audio, sr = librosa.load(
        file,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0).reshape(1,-1)

st.title("🎤 Speech Emotion Recognition")

uploaded = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded:

    st.audio(uploaded)

    features = extract_features(uploaded)

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    emotion = encoder.inverse_transform(prediction)[0]

    confidence = np.max(probability) * 100

    st.success(f"Predicted Emotion: {emotion.upper()}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("All Emotion Probabilities")

    for emo, prob in zip(encoder.classes_, probability[0]):
        st.write(f"{emo}: {prob*100:.2f}%")