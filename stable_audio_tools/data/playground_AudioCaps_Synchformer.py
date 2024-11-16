import json
import os
import pickle
import csv

def csv_to_json_dict(csv_file_path, json_file_path):
    
    """
    This function loads a csv file, creates a dictionary with video IDs as keys and prompts, start_sec and caption as values,
    and saves the dictionary to a JSON file.

    Args:
    - json_file_path (str): The path to the input JSON file.
    - output_file_path (str): The path to save the output JSON file containing the prompts dictionary.

    Returns:
    - dict: The dictionary containing video IDs as keys and prompts as values.
    """
    data_dict = {}

    # Open and read the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        # Loop through each row and construct the dictionary
        for row in csv_reader:
            audiocap_id = row['audiocap_id']
            data_dict[audiocap_id] = {
                'youtube_id': row['youtube_id'],
                'start_time': int(row['start_time']),
                'caption': row['caption']
            }

    # Write the dictionary to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, indent=4, ensure_ascii=False)

    print(f"Data successfully saved to {json_file_path}")
    return data_dict


# def create_and_save_prompts_dict(json_file_path, output_file_path):
#     """
#     This function loads a JSON file, creates a dictionary with video IDs as keys and prompts as values,
#     and saves the dictionary to a JSON file.

#     Args:
#     - json_file_path (str): The path to the input JSON file.
#     - output_file_path (str): The path to save the output JSON file containing the prompts dictionary.

#     Returns:
#     - dict: The dictionary containing video IDs as keys and prompts as values.
#     """
#     # Load the JSON file
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)

#     # Create a dictionary where each key is a video ID and the value is the prompt
#     prompts_dict = {}
#     for instrument, video_ids in data["videos"].items():
#         for video_id in video_ids:
#             # prompts_dict[video_id] = f"Prompt for {instrument} video {video_id}"  # Modify the prompt format as needed
#             prompts_dict[video_id] = instrument

#     # Save the dictionary to a JSON file
#     with open(output_file_path, 'w') as f:
#         json.dump(prompts_dict, f, indent=4)

#     # Return the dictionary
#     return prompts_dict



def create_final_dict(root_directory, Synchformer_embed_root_directory, prompts_dict):
    """
    This function iterates over a root directory, matches mp3 files with video IDs, and creates a dictionary
    where each video ID maps to a list containing a dictionary with prompt and Synchformer embed file path.

    Args:
    - root_directory (str): The path to the root directory containing mp3 files.
    - Synchformer_embed_directory (str): The path to the directory containing Synchformer embed files.
    - prompts_dict (dict): A dictionary containing video IDs as keys and prompts as values.

    Returns:
    - dict: The final dictionary with video IDs as keys and a list of dictionaries with prompts and Synchformer embed paths as values.
    """
    final_dict = {}

    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".mp3"):
                file_name = os.path.splitext(file)[0]
                Audio_caps_id = os.path.splitext(file)[0].split('_')[-1]  # Get the video ID from the file name
                cur_Info_Dict = prompts_dict.get(Audio_caps_id, "No video_id available")

                prompt = cur_Info_Dict.get("caption", "No prompt available")
                Video_ID = cur_Info_Dict.get("youtube_id", "No prompt available")
                # Find the corresponding .npy file in the npy_root_directory
                relative_subdir = os.path.relpath(subdir, root_directory)  # Get relative subdirectory path
                Synchformer_filename = Video_ID + "_" + Audio_caps_id + ".npy"
                npy_Synchformer_file_path = os.path.join(Synchformer_embed_root_directory, Synchformer_filename)
                
                # Check if the .npy file exists
                if not os.path.exists(npy_Synchformer_file_path):
                    raise FileNotFoundError(f"Missing .npy file for {Audio_caps_id}: {npy_Synchformer_file_path}")
                
                else:
                    final_dict[file_name] = {
                        "track_file_name": prompt,
                        "Synchformer_file_path": npy_Synchformer_file_path,
                    }

                    


    return final_dict

def process_videos(csv_file_path, output_prompts_file, root_directory, Synchformer_embed_directory, output_file_path):
    """
    This function combines the two processes: creating the prompts dictionary and creating the final dictionary
    based on the root directory, Synchformer embed directory, and the generated prompts dictionary.

    Args:
    - json_file_path (str): The path to the input JSON file.
    - output_prompts_file (str): The path to save the output JSON file containing the prompts dictionary.
    - root_directory (str): The path to the root directory containing mp3 files.
    - Synchformer_embed_directory (str): The path to the directory containing Synchformer embed files.

    Returns:
    - dict: The final dictionary with video IDs as keys and a list of dictionaries with prompts and Synchformer embed paths as values.
    """
    # Create and save the prompts dictionary

    prompts_dict = csv_to_json_dict(csv_file_path, output_prompts_file)
    # print(prompts_dict)
    # Create the final dictionary using the prompts dictionary
    final_dict = create_final_dict(root_directory, Synchformer_embed_directory, prompts_dict)
    

        
    # Save the dictionary to a file
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_dict, f)
        
        
    return final_dict

data_set_type = "small_test"
# Usage example Unnormelized clips
csv_file_path = f'/home/rbenita/Projects/stable-audio-tools-synchformer-cond-216-emb/Subjective_Evaluation/metadata_files/{data_set_type}.csv'
output_prompts_file = f'Subjective_Evaluation/metadata_json/AudioCaps_Metadata_dict_{data_set_type}_synchformer.json'
root_directory = f"Subjective_Evaluation/{data_set_type}/short_audio_files/all_files"
# root_directory = f'/home/rbenita/Dataset_Synchformer/{data_set_type}_2/short_audio_files'
clip_embed_directory = f'Subjective_Evaluation/{data_set_type}/synchformer_npy_padded_1_216_768/single_object_movements'
output_file_path = f"Subjective_Evaluation/pkl_files/{data_set_type}_synchformer_npy_padded_1_216_768.pkl"


final_dict = process_videos(csv_file_path, output_prompts_file, root_directory, clip_embed_directory, output_file_path)
print(final_dict)
# The final_dict now contains the desired structure.

