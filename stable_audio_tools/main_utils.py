from pedalboard.io import AudioFile
from torchaudio import transforms as T
import torch
import torchaudio
import os 

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



def get_file_name_without_extension(relative_path):
    # Extract the base name (file name with extension) from the path
    base_name = os.path.basename(relative_path)
    # Split the base name into name and extension and return the name part
    file_name_without_extension = os.path.splitext(base_name)[0]
     # Remove leading zeros
    file_name_cleaned = file_name_without_extension.lstrip('0')
    return file_name_cleaned

def get_file_name_clip_implementation(relative_path):
    # Extract the base name (file name with extension) from the path
    base_name = os.path.basename(relative_path)
    # Split the base name into name and extension and return the name part
    file_name_without_extension = os.path.splitext(base_name)[0]
     # Remove leading zeros
    # file_name_cleaned = file_name_without_extension.lstrip('0')
    return file_name_without_extension
