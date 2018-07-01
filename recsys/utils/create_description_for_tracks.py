import sys
import json
from tqdm import tqdm
import utils.load_info_for_model as load_info_for_model

def prepare_name(name, prefix):
    name = name.lower()
    name = name.split()
    name = [prefix + x  for x in name]
    return " ".join(name)



def album_name(track):
    return album_names[track_to_album[track]]

def artist_name(track):
    return artist_names[album_to_artist[track_to_album[track]]]

def track_name(track):
    return track_names[track]

if __name__ == "__main__":
    track_to_album = load_info_for_model.load_json(sys.argv[1])
    album_to_artist = load_info_for_model.load_json(sys.argv[2])
    track_names = load_info_for_model.load_json(sys.argv[3])
    album_names= load_info_for_model.load_json(sys.argv[4])
    artist_names= load_info_for_model.load_json(sys.argv[5])
    output_filename = sys.argv[6]
    max_track = len(track_names)
    with open(output_filename, "w") as output_file:
        for i in tqdm(range(max_track)):
            name = prepare_name(" ".join([
                prepare_name(artist_name(i), "__artist__"),
                prepare_name(album_name(i), "__album__"),
                prepare_name(track_name(i), "__track__"),
            ]), "__label__")
            output_file.write(name + "\n")
