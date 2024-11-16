import torch
import numpy as np
# from audiocraft.models import musicgen
# from audiocraft.utils.notebook import display_audio
import torch
import os
# os.system("python -m pip install -U git+https://github.com/facebookresearch/audiocraft#egg=audiocraft")
import pickle
from tqdm import tqdm
import yaml
print("hello this is inference")

from huggingface_hub import login
# Replace 'your_token_here' with the token you generated
login(token="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")


import warnings

# Suppress only FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import re
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools.models.pretrained import get_pretrained_model, get_pretrained_model_local
from stable_audio_tools.inference.generation import generate_diffusion_cond
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# Download model
# model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model_name = "piowpufq"
from main_utils import load_file, get_file_name_clip_implementation
root_models_path = f"/home/rbenita/Projects/stable-audio-tools-synchformer-cond/save_dir/StableAudioOpen_with_ControlNet/{model_name}/checkpoints"

# Regular expression pattern to match the epoch and step numbers
# Example filename: epoch=31-step=4000.ckpt
pattern = re.compile(r'epoch=(\d+)-step=(\d+)\.ckpt$')
control_type = "test"
control_files_path = f"/home/rbenita/Projects/stable-audio-tools-synchformer-cond/Subjective_Evaluation/{control_type}/short_audio_files/short_audio_by_tests"
prompts_dict_path = f"/home/rbenita/Projects/stable-audio-tools-synchformer-cond/Subjective_Evaluation/pkl_files/dict_clip_AudioCaps_Synchformer_subjective_eval_test.pkl"
text_by_complexity_path = "/home/rbenita/Projects/stable-audio-tools-synchformer-cond/Subjective_Evaluation/test/Levels_of_text_complexity/Texts_by_complexity.yaml"
wanted_steps = [20000]
cfg_scale_array = [7]
seeds = [10,20,30]
len = 10
prompts = []


# Load the YAML file
with open(text_by_complexity_path, 'r') as file:
    text_by_complexity_data = yaml.safe_load(file)
    
# Iterate over each file in the folder
for filename in os.listdir(root_models_path):
    match = pattern.search(filename)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        
        if not(step in wanted_steps):
            print(f"dont need step: {step}")
            continue
        
        print(f"File: {filename}, Epoch: {epoch}, Step: {step}")

        for cur_cfg_scale in cfg_scale_array:
        
            model_config_path = "/home/rbenita/Projects/stable-audio-tools-synchformer-cond/stable-audio-tools/stable_audio_tools/csa/csa_model_config_controlnet_Synchformer.json"
            # model_ckpt_path = "/home/rbenita/Projects/Text_to_audio_with_controlnet/stable_audio_open_controlnet/stable-audio-tools-intensity-cond/save_dir/Intensity_1/StableAudioOpen_with_ControlNet/g7y30faq/checkpoints/epoch=10-step=10000.ckpt"
            target_path = f"/home/rbenita/Projects/stable-audio-tools-synchformer-cond/Subjective_Evaluation/{control_type}/results/{model_name}/generated_audio/step_{step}/cfg_{cur_cfg_scale}"
            model_ckpt_path = f"/home/rbenita/Projects/stable-audio-tools-synchformer-cond/save_dir/StableAudioOpen_with_ControlNet/{model_name}/checkpoints/epoch={epoch}-step={step}.ckpt"

            # Ensure the target directory exists
            os.makedirs(target_path, exist_ok=True)

            if prompts_dict_path is not None:
                with open(prompts_dict_path, 'rb') as f:
                    prompts_dict = pickle.load(f)
                    
                    
            model, model_config = get_pretrained_model_local(model_config_path, model_ckpt_path)
            sample_rate = model_config["sample_rate"]
            sample_size = model_config["sample_size"]

            model = model.to(device)
            model.eval()
        
            # for filename in tqdm(os.listdir(control_files_path)):
            #     file_path = os.path.join(control_files_path, filename)
            #     if file_path.lower().endswith('.mp3'):
                    
            for subdir, _, files in os.walk(control_files_path):
                for file in files:
                    if file.endswith(".mp3"):  # Only process .mp3 files
                        file_path = os.path.join(subdir, file)
                        
                        _, in_sr = load_file(file_path, sample_rate)
                        prompt = prompts_dict[get_file_name_clip_implementation(file_path)]["track_file_name"]
                        ##
                        synchformer_file_path = prompts_dict[get_file_name_clip_implementation(file_path)]["Synchformer_file_path"]
                        print(f"the synchformer path :{synchformer_file_path}")
                        synchformer_embedding =torch.from_numpy(np.load(prompts_dict[get_file_name_clip_implementation(file_path)]["Synchformer_file_path"]))
                        synchformer_embedding = synchformer_embedding.to(device)
                        
                        # cur_prompt = prompt
                        prompts.append(prompt)
                        # Create corresponding subdirectory in the target directory
                        relative_subdir = os.path.relpath(subdir, control_files_path)
                        output_subdir = os.path.join(target_path, relative_subdir)
                        
                        # Ensure the output subdirectory exists
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        if "text_complexity_audios" in output_subdir:
                            prompts = text_by_complexity_data[file[:-4]]
                            
                        for cur_seed in seeds:
                            for idx, cur_prompt in enumerate(prompts):
                                info = f"scale={cur_cfg_scale}_prompt_{idx}={cur_prompt}_seed={cur_seed}"

                                conditioning = [{
                                    "Synchformer_signal": synchformer_embedding,
                                    "prompt": cur_prompt,
                                    "seconds_start": 0, 
                                    "seconds_total": len
                                }]

                                # Generate stereo audio
                                output = generate_diffusion_cond(
                                    model,
                                    steps=100,
                                    # cfg_scale=7,
                                    cfg_scale=cur_cfg_scale,
                                    seed = cur_seed,
                                    conditioning=conditioning,
                                    sample_size=sample_size,
                                    ## Intensity_ change: 
                                    sample_rate=sample_rate,
                                    sigma_min=0.3,
                                    sigma_max=500,
                                    sampler_type="dpmpp-3m-sde",
                                    device=device
                                )

                                # Rearrange audio batch to a single sequence
                                output = rearrange(output, "b d n -> d (b n)")
                                # Generate the output file path
                                output_file_path = os.path.join(output_subdir, f"filename_{file[:-4]}_{info}_len_{len}.mp3")
                                # Peak normalize, clip, convert to int16, and save to file
                                output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                                
                                torchaudio.save(output_file_path, output, sample_rate)
