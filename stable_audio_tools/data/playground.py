import csv
import os
import pickle

def create_track_dict(csv_file, output_file):
    track_dict = {}
    
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            track_id = row['track_id']
            track_file = row['track_file']
            # Extract the last part of the track file after the last '/'
            last_part = os.path.basename(track_file)
            # Remove the file extension
            last_part_no_ext = os.path.splitext(last_part)[0]
            track_dict[track_id] = last_part_no_ext

    # Save the dictionary to a file
    with open(output_file, 'wb') as f:
        pickle.dump(track_dict, f)


if __name__ == "__main__":

    # Example usage
    csv_file = '/home/rbenita/Projects/Text_to_audio_with_controlnet/stable_audio_open_controlnet/datasets/fma_small/metadata/fma_metadata/raw_tracks.csv'
    output_file = '/home/rbenita/Projects/Text_to_audio_with_controlnet/stable_audio_open_controlnet/datasets/fma_small/track_dict.pkl'
    create_track_dict(csv_file, output_file)

    # To load the dictionary later, you can use the following code:
    with open(output_file, 'rb') as f:
        loaded_dict = pickle.load(f)
        print(loaded_dict)
