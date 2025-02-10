import os
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
import librosa


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def pySTFT(x, fft_length=1920, hop_length=480):
    x = np.pad(x, int(fft_length//2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    

mel_basis = librosa.filters.mel(sr=48000, n_fft=1920, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 48000, order=5)

# Audio file directory
rootDir = r'C:\Users\DELL\OneDrive\Desktop\Internship\ADESH'
# Spectrogram directory
targetDir = r'C:\Users\DELL\OneDrive\Desktop\Internship\mel'

# Get list of .wav files directly from rootDir
fileList = [f for f in os.listdir(rootDir) if f.endswith('.wav')]

for fileName in sorted(fileList):
    file_path = os.path.join(rootDir, fileName)

    try:
        # Read audio file
        x, fs = sf.read(file_path)

        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        
        # Add random noise for model robustness
        prng = RandomState(abs(hash(fileName)) % (2**32 - 1))  # Use the hash of the filename as a seed
        wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
        
        # Compute spectrogram
        D = pySTFT(wav).T

        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)    

        # Save spectrogram
        output_file_path = os.path.join(targetDir, fileName[:-4] + '.npy')
        np.save(output_file_path, S.astype(np.float32), allow_pickle=False)
    
    except:
        # Skip any unreadable files without showing an error
        continue
