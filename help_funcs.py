#importing packages
import numpy as np
import base64
import librosa 
import numpy as np
import pandas as pd
#function to convert base64 to audio

#function to convert base64 to .wav-file

def base64_decode_audio(directory_path,id, encode_string):
    #real file path was changed due to confidentiality
    wav_file= open((directory_path + id + '.wav'), "wb")
    decode_string = base64.b64decode(encode_string)
    wav_file.write(decode_string)

#function to extract features for each sound file

def extract_features(file):

    # loads the audio file as a floating point time series and assigns the default sample rate
    # sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    result = np.array([])

    # generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    result=np.hstack((result, mfccs))

    # generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result=np.hstack((result, chroma))

    # computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    result=np.hstack((result, mel))

    # computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    result=np.hstack((result, contrast))

    # computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    result=np.hstack((result, tonnetz))

    return result