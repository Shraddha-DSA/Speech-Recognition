import librosa
import numpy as np
def extract_features(file):
    audio,sr = librosa.load(file,sr=16000)
    audio,_ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)
    mfcc = np.mean(librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13),axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio),axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio,sr=sr),axis=1)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio,sr=sr),axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio,sr=sr),axis=1)
    features=np.hstack([mfcc,zcr,centroid,bandwidth,rolloff])
    return features