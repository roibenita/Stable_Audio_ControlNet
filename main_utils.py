from pedalboard.io import AudioFile
from torchaudio import transforms as T
import torch
import torchaudio

def load_file(filename, model_sr):
    ext = filename.split(".")[-1]

    if ext == "mp3":
        with AudioFile(filename) as f:
            audio = f.read(f.frames)
            audio = torch.from_numpy(audio)
            in_sr = f.samplerate
    else:
        audio, in_sr = torchaudio.load(filename, format=ext)

    if in_sr != model_sr:
        resample_tf = T.Resample(in_sr, model_sr)
        audio = resample_tf(audio)

    return audio, in_sr

