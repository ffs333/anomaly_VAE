import librosa
import librosa.feature
import librosa.display
import numpy as np
import soundfile as sf
import pylab


def paint(waveform, path, epochs, typ, sp, sd):
    """
    Create and save spectrogram plot
    :param waveform: (numpy ndarray) spectrogram
    :param path: (str) path to save plot
    :param epochs: (int) epochs number
    :param typ: (str) type of process. train/eval
    :param sp: (str) type of spectrogram. mfcc/mel
    :param sd: (str) input or reconstruction
    """
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge

    librosa.display.specshow(librosa.power_to_db(waveform, ref=np.max))  # mfcc spec
    pylab.savefig(path + f'/{epochs}_{typ}_{sp}_{sd}.jpg', bbox_inches=None, pad_inches=0)
    pylab.close()


def draw(inp, recons, epoch, step, out_path, ev=True):
    """
    Paint mfcc and mel spectrogram plots and save inverse audios
    :param inp: (Tensor) input spectrogram
    :param recons: (Tensor) reconstructed spectrogram
    :param epoch: (int) epochs number
    :param step: (int) step to create mel spectrogram
    :param out_path: (str) path to save images and audio
    :param ev: (bool) type of process. train/eval
    """
    hz = 'eval' if ev else 'train'

    inp = inp.cpu().numpy()
    inp = inp[0]
    recons = recons.cpu().detach().numpy()
    recons = recons[0]

    paint(inp, out_path, epoch, hz, 'mfcc', 'in')
    paint(recons, out_path, epoch, hz, 'mfcc', 'rec')

    if epoch % step == 0:
        audio_in = librosa.feature.inverse.mfcc_to_audio(inp)
        audio_rec = librosa.feature.inverse.mfcc_to_audio(recons)

        sf.write(out_path + f'/{epoch}_{hz}_audio_in.wav', audio_in, 48000)
        sf.write(out_path + f'/{epoch}_{hz}_audio_rec.wav', audio_rec, 48000)

        mel_in = librosa.feature.melspectrogram(y=audio_in, sr=48000)
        mel_rec = librosa.feature.melspectrogram(y=audio_rec, sr=48000)

        paint(mel_in, out_path, epoch, hz, 'mel', 'in')
        paint(mel_rec, out_path, epoch, hz, 'mel', 'rec')