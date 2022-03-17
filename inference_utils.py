import math

from librosa import power_to_db, stft
import numpy as np


def mel_bank(n_mels=256):
    n_freqs = 1025
    f_min = 0.0
    f_max = 24000.0
    sample_rate = 48000

    all_freqs = np.linspace(0, sample_rate // 2, n_freqs).astype('float32')

    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = np.linspace(m_min, m_max, n_mels + 2).astype('float32')
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = np.expand_dims(f_pts, axis=0).astype('float32') - np.expand_dims(all_freqs, axis=1).astype('float32')

    zero = np.zeros(1).astype('float32')
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = np.maximum(zero, np.minimum(down_slopes, up_slopes).astype('float32')).astype('float32')
    return fb


def create_dct(n_mfcc=32, n_mels=256):
    n = np.arange(float(n_mels))
    k = np.expand_dims(np.arange(float(n_mfcc)), axis=1)
    dct = np.cos(math.pi / float(n_mels) * (n + 0.5) * k).astype('float32')
    dct[0] *= 1.0 / math.sqrt(2.0)
    dct *= math.sqrt(2.0 / float(n_mels))
    return dct.T


def make_spec(x, mel=True):
    x = stft(x, n_fft=2048, hop_length=563, win_length=2048, center=True)

    x = np.abs(x) ** 2.0
    if mel:
        x = np.dot(np.transpose(x, (1, 0)), mel_bank())
        x = np.transpose(x, (1, 0))

    return x


def make_batch(x):

    x = power_to_db(x)
    dct_mat = create_dct()
    x = np.dot(np.transpose(x, (-1, 0)), dct_mat)
    x = np.transpose(x, (-1, 0))
    return np.expand_dims(x, axis=(1, 0))


def numpy_mse(array1, array2):
    difference_array = np.subtract(array1, array2)
    squared_array = np.square(difference_array)
    return squared_array.mean()


def get_percent2(val, threshold=130, multi=4):
    if val <= threshold:
        return 0
    if threshold < val <= threshold * multi:
        one_percent = 1 / (threshold* multi - threshold)
        percent = (val - threshold) * one_percent
        return percent*100
    if val > threshold * multi:
        return 100