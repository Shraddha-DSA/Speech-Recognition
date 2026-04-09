import os
import tensorflow as tf
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
dataset_path = r"C:\Users\Shraddha\Desktop\speech_emotion_recognition\dataset\RAVDESS"
X = []
y = []
emotion_map = {
    "01":"neutral",
    "02":"calm",
    "03":"happy",
    "04":"sad",
    "05":"angry",
    "06":"fearful",
    "07":"disgust",
    "08":"surprised"
}
print("Loading Dataset")
for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path,actor)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                emotion = emotion_map[emotion_code]
                file_path = os.path.join(actor_path,file)
                audio,sr = librosa.load(file_path,sr=16000)
                mel = librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=128)
                mel_db = librosa.power_to_db(mel)
                if mel_db.shape[1]<128:
                    pad = 128 - mel_db.shape[1]
                    mel_db = np.pad(mel_db,((0,0),(0,pad)),mode="constant")
                else:
                    mel_db = mel_db[:,:128]
                X.append(mel_db)
                y.append(emotion_map[emotion_code])
print("Dataset loaded")
X = np.array(X)
y = np.array(y)
X = X[...,np.newaxis]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))
y = tf.keras.utils.to_categorical(y,num_classes)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.3),
    Dense(num_classes,activation='softmax')
])
model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=20,batch_size=32,validation_data=(X_test,y_test))
model.save("cnn_emotion_model.h5")
np.save("label_encoder.npy",le.classes_)
print("Model saved")