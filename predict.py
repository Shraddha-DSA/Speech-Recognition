import numpy as np
import tensorflow as tf
import librosa
from feature_extraction import extract_features
model = tf.keras.models.load_model("cnn_emotion_model.h5")
labels  = np.load("label_encoder.npy")
file = r"C:\Users\Shraddha\Downloads\nand1nho-happy-smile-465402.mp3"
audio, sr = librosa.load(file, sr=16000)

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(mel)

if mel_db.shape[1] < 128:
    pad = 128 - mel_db.shape[1]
    mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode='constant')
else:
    mel_db = mel_db[:, :128]

mel_db = mel_db[np.newaxis, ..., np.newaxis]

prediction = model.predict(mel_db)

print("Predicted Emotion:", labels[np.argmax(prediction)])