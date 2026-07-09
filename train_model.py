import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET_PATH = "dataset"

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)


X = []
y = []

print("Loading Dataset...")

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            emotion = emotion_dict[file.split("-")[2]]

            path = os.path.join(actor_path, file)

            X.append(extract_features(path))

            y.append(emotion)

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))

encoder = LabelEncoder()

y = encoder.fit_transform(y)

joblib.dump(encoder, "label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training SVM...")

model = SVC(
    kernel="rbf",
    probability=True
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy * 100)

joblib.dump(model, "emotion_model.pkl")

print("Model Saved Successfully!")