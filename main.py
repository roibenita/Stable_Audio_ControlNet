import torch
import numpy as np
# from audiocraft.models import musicgen
# from audiocraft.utils.notebook import display_audio
import torch
import os
# os.system("python -m pip install -U git+https://github.com/facebookresearch/audiocraft#egg=audiocraft")


print("hello")

from huggingface_hub import login
# Replace 'your_token_here' with the token you generated
login(token="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")


import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model, get_pretrained_model_local
from stable_audio_tools.inference.generation import generate_diffusion_cond
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# Download model
# model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")

from main_utils import load_file

model_config_path = "/home/rbenita/Projects/Text_to_audio_with_controlnet/stable_audio_open_controlnet/stable-audio-tools/stable_audio_tools/csa/csa_model_config_controlnet.json"
model_ckpt_path = "/home/rbenita/Projects/Text_to_audio_with_controlnet/stable_audio_open_controlnet/stable-audio-tools-try-same-audio/stable-audio-tools/stable_audio_tools/csa/save_dir/StableAudioOpen_with_ControlNet/hzl5i5rk/checkpoints/epoch=4999-step=5000.ckpt"

model, model_config = get_pretrained_model_local(model_config_path, model_ckpt_path)


sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
# torch.set_default_tensor_type(torch.cuda.HalfTensor)

model = model.to(device)
# Set up the length of the synthesis (in seconds)
# Set up text and timing conditioning

audio_path = "/home/rbenita/Projects/Text_to_audio_with_controlnet/stable_audio_open_controlnet/datasets/fma_small/dataset/fma_small/155/155066.mp3"

audio, in_sr = load_file(audio_path, sample_rate)

##TODO: Preprocess the signal is needed.

conditioning = [{
    "control_signal": audio,
    "prompt": "JBlanked_-_03_-_Roy",
    "seconds_start": 0, 
    "seconds_total": 15
}]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sample_rate=in_sr,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)
