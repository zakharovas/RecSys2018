import sys
import json
from tqdm import tqdm
import re

import utils.load_info_for_model as load_info_for_model


def prepare_name(name, prefix):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.split()
    name = [prefix + x for x in name]
    return " ".join(name)


def album_name(track):
    return album_names[track_to_album[track]]


def artist_name(track):
    return artist_names[album_to_artist[track_to_album[track]]]


def track_name(track):
    return track_names[track]


if __name__ == "__main__":
    print(sys.argv)
    playlists_filename = sys.argv[1]
    track_to_album = load_info_for_model.load_json(sys.argv[2])
    album_to_artist = load_info_for_model.load_json(sys.argv[3])
    track_names = load_info_for_model.load_json(sys.argv[4])
    album_names = load_info_for_model.load_json(sys.argv[5])
    artist_names = load_info_for_model.load_json(sys.argv[6])
    output_filename = sys.argv[7]
    mode = int(sys.argv[8])
    with open(playlists_filename) as playlist_file:
        playlists = [json.loads(line) for line in tqdm(playlist_file)]
    with open(output_filename, "w") as output_file:
        for playlist in tqdm(playlists):
            names = []
            if mode == 1:
                if "name" not in playlist or playlist["name"] == "" or playlist["name"] == "#no-name#" or len(
                        playlist["tracks"]) == 0:
                    continue
                names = [prepare_name(playlist["name"], "")]
                for track in playlist["tracks"]:
                    names.append(prepare_name(" ".join([
                        prepare_name(artist_name(track), "__artist__"),
                        prepare_name(album_name(track), "__album__"),
                        prepare_name(track_name(track), "__track__"),
                    ]), "__label__"))
            elif mode == 0:
                names = [prepare_name(playlist["name"], "")]
            output_file.write("\t".join(names) + "\n")
