import io

import numpy as np
from scipy.io.wavfile import read

from inference_utils import make_spec, make_batch, get_percent2, numpy_mse

cur_threshold = 130


def pipeline(model, input_wav, classifier=None, secs=6):

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

    spec = make_spec(audio, mel=True)
    mfcc = make_batch(spec)

    model_out = model.run(None, {'input_1': mfcc})
    mse_val = numpy_mse(mfcc, model_out[0])
    percent = get_percent2(mse_val, cur_threshold, 5)
    if classifier:
        class_out = np.argmax(classifier.run(None, {'input_1': mfcc}))
        return mse_val, percent, class_out.item()
    return mse_val, percent

