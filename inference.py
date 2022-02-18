import io

import numpy as np
from scipy.io.wavfile import read

from inference_utils import make_mel, make_batch, get_percent2, numpy_mse

cur_threshold = 130


def pipeline(model, input_wav, classifier=None, secs=3):

    sr, audio = read(io.BytesIO(input_wav))
    if audio.dtype != 'float32':
        audio = audio.astype(np.float32, order='C') / 32768.0
    if sr == 48000:
        # 0 for left channel, 1 for right
        if audio.ndim == 1:
            audio = audio[:int(sr*secs)]
        else:
            audio = audio[:int(sr*secs), 0]
    else:
        print(f'Invalid sample rate: {sr}. 48000 needed')

    volume_met = [abs(audio).max().item(), np.nonzero(abs(audio) > 0.4)[0].shape[0]]
    if volume_met[0] != 1 and volume_met[1] < 5000:
        print("Engine turned off. Silence in audio.")
        return -1, -1, 0

    spec = make_batch(audio)

    model_out = model.run(None, {'input_1': spec})
    mse_val = numpy_mse(spec, model_out[0])
    percent = get_percent2(mse_val, cur_threshold)
    if classifier:
        class_out = classifier.run(None, {'input_1': spec})
        if class_out[0].item() < 0.5:
            return -1, -1, 0
        return mse_val, percent, round(class_out[0].item())
    return mse_val, percent

