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
    line.append("|gName")
    line.append(str(name_encoding[normalize_name(name)]))
    return " ".join(line)


if __name__ == "__main__":
    train_file = sys.argv[1]
    track_to_album_file = sys.argv[2]
    album_to_artis_file = sys.argv[3]
    name_encoding_file = sys.argv[4]
    output_filename = sys.argv[5]
    track_to_album = np.array(utils.load_info_for_model.load_json(track_to_album_file))
    album_to_artist = np.array(utils.load_info_for_model.load_json(album_to_artis_file))
    name_encoding = utils.load_info_for_model.load_json(name_encoding_file)
    with open(train_file) as train, open(output_filename, "w") as output_file:
        for line in tqdm(train):
            playlist = json.loads(line)
            tracks = set(playlist["tracks"])
            sizes = np.random.permutation([1, 5, 10, 25, 100])
            remain_size = 0
            for cur_size in sizes:
                if len(tracks) > cur_size + 5:
                    remain_size = cur_size
                    break
            if remain_size == 0:
                continue
            positions = set(np.random.permutation(np.arange(len(playlist["tracks"])))[:remain_size])
            remained_tracks = [x for j, x in enumerate(playlist["tracks"]) if j in positions]
            deleted_tracks = [x for j, x in enumerate(playlist["tracks"]) if j not in positions]
            random_tracks = np.random.randint(0, len(track_to_album), 50)

            for track in deleted_tracks:
                output_file.write(create_line(remained_tracks, playlist["name"], track, 1) + "\n")
            for track in random_tracks:
                if track not in tracks:
                    output_file.write(create_line(remained_tracks, playlist["name"], track, -1) + "\n")
