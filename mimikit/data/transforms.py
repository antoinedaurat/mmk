import librosa
from .metadata import Metadata
import numpy as np

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
MU = 255


def stft_concat_channels(file, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    y, sr = librosa.load(file, mono=False, sr=sr)
    ffts = [librosa.stft(y[k], n_fft=n_fft, hop_length=hop_length) for k in range(y.shape[0])]
    ffts = np.concatenate(ffts, 0)
    # returns the feature and its attributes
    attrs = dict(n_fft=n_fft, 
                 hop_length=hop_length, 
                 sr=sr,
                 stft_implementation='librosa',
                 center=True,
                 window='hann',
                 channels=y.shape[0],
                 win_length=n_fft,
                 normalization='none',
                 zp_window=False,
                 )
    return ffts, attrs


def multichannel_file_to_fft(abs_path, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    fft, params = stft_concat_channels(abs_path, n_fft, hop_length, sr)
    fft = abs(fft)
    metadata = Metadata.from_duration([fft.shape[1]])
    params.update(dict(time_axis=0))
    return dict(fft=(params, fft.T), metadata=({}, metadata))


def stft(file, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    y, sr = librosa.load(file, sr=sr)
    fft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # returns the feature and its attributes
    attrs = dict(n_fft=n_fft, 
                 hop_length=hop_length, 
                 sr=sr,
                 stft_implementation='librosa',
                 center=True,
                 window='hann',
                 win_length=n_fft,
                 normalization='none',
                 zp_window=False,
                 )
    return fft, attrs


def file_to_fft(abs_path, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    fft, params = stft(abs_path, n_fft, hop_length, sr)
    fft = abs(fft)
    metadata = Metadata.from_duration([fft.shape[1]])
    params.update(dict(time_axis=0))
    return dict(fft=(params, fft.T), metadata=({}, metadata))


def mu_law_compress(file, mu=MU, sr=SR):
    y, sr = librosa.load(file, sr=sr)
    y = librosa.util.normalize(y)
    qx = librosa.mu_compress(y, mu, quantize=True)
    qx = qx + (MU + 1) // 2
    return qx, dict(mu=mu, sr=sr)


def file_to_qx(abs_path, mu=MU, sr=SR):
    qx, params = mu_law_compress(abs_path, mu, sr)
    metadata = Metadata.from_duration([qx.shape[0]])
    return dict(qx=(params, qx.reshape(-1, 1)), metadata=({}, metadata))


default_extract_func = file_to_fft
