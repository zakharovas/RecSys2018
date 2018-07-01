import json
from tqdm import tqdm
import sys
import numpy as np
import utils.load_info_for_model
import re


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def create_line(remained, name, candidate, target):
    line = []
    line.append(str(target))
    line.append("|aArtistPl")
    line += [str(album_to_artist[track_to_album[x]]) for x in remained]
    line.append("|bAlbumPl")
    line += [str(track_to_album[x]) for x in remained]
    line.append("|cTrackPl")
    line += [str(x) for x in remained]
    line.append("|dArtistCand")
    line.append(str(album_to_artist[track_to_album[candidate]]))

    line.append("|eAlbumCand")
    line.append(str(track_to_album[candidate]))
    line.append("|fTrackCand")
    line.append(str(candidate))
    if with_name > 0 and len(name) > 0 and name != "#no-name#" and normalize_name(name) in name_encoding:
        line.append("|gName")
        line.append(str(name_encoding[normalize_name(name)]))
    return " ".join(line)


if __name__ == "__main__":
    train_file = sys.argv[1]
    track_to_album_file = sys.argv[2]
    album_to_artis_file = sys.argv[3]
    name_encoding_file = sys.argv[4]
    output_filename = sys.argv[5]
    with_name = int(sys.argv[6])
    track_to_album = np.array(utils.load_info_for_model.load_json(track_to_album_file))
    album_to_artist = np.array(utils.load_info_for_model.load_json(album_to_artis_file))
    name_encoding = utils.load_info_for_model.load_json(name_encoding_file)
    with open(train_file) as train, open(output_filename, "w") as output_file:
        for line in tqdm(train):
            playlist = json.loads(line)
            remained_tracks = playlist["tracks"]
            deleted_tracks = [x[0] for x in playlist["candidates"][0][2]]
            target = playlist["target"]

            for track, trgt in zip(deleted_tracks, target):

                if "name" in playlist and len(remained_tracks) < 25:
                    name = playlist["name"]
                else:
                    name = ""
                output_file.write(create_line(remained_tracks, name, track, trgt) + "\n")

