The program is designed to process audio recordings in order to classify them to detect the answering machine. The processing is based on the analysis of background noise by extracting of various features from the audio(MFCCs, STFT, chromagram, mel-scaled spectrogram, spectral contrast, tonnetz).
Audio recordings are sent from IP telephony to the Redis queue in Base64 format and stored under the corresponding unique index (ID). The program, connecting to Redis, receives audio recordings from the queue and processes them sequentially.
During the processing of an audio recording using a prepaid logistic regression model, a forecast is made whether an answering machine is present in the recording or not. 
The processing result, which is a forecast generated by the model, is returned to the Redis queue by the audio recording ID.
