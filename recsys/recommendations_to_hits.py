import collections
import sys
import json
from tqdm import tqdm
import numpy as np

import utils.load_info_for_model


def recommendations_to_hits(recommended, correct, track_to_album, album_to_artist):
    track_size = track_to_album.size
    artist_size = np.max(album_to_artist) + 1
    artist_recommended = album_to_artist[track_to_album[recommended]]
    artist_correct = album_to_artist[track_to_album[correct]]

    track_stats = np.zeros(track_size)
    count = np.bincount(correct)
    track_stats[:count.size] = count

    artist_stats = np.zeros(artist_size)
    count = np.bincount(artist_correct)
    artist_stats[:count.size] = count
    track_vector = []
    artist_vector = []

    for candidate_track, candidate_artist in zip(recommended, artist_recommended):
        correct_track = False
        if track_stats[candidate_track] > 0:
            correct_track = True
            track_stats[candidate_track] -= 1
        correct_artist = False
        if artist_stats[candidate_artist] > 0:
            correct_artist = True
            artist_stats[candidate_artist] -= 1
        track_vector.append(int(correct_track))
        artist_vector.append(int(correct_artist))
    return track_vector, artist_vector


if __name__ == '__main__':
    playlists = utils.load_info_for_model.load_lines_json(sys.argv[1])
    print("TEST READ")
    recommendations_filename = sys.argv[2]
    track_to_album = sys.argv[3]
    album_to_artist = sys.argv[4]
    output_filename = sys.argv[5]
    track_to_album = np.array(utils.load_info_for_model.load_json(track_to_album))
    album_to_artist = np.array(utils.load_info_for_model.load_json(album_to_artist))

    test_dict = dict()
    for playlist in playlists:
        test_dict[playlist["pid"]] = playlist["deleted_track"]
    with open(recommendations_filename) as recommendations_file, open(output_filename, "w") as output_file:
        for i, playlist in enumerate(tqdm(recommendations_file)):
            playlist = json.loads(playlist)
            result = dict()
            result["pid"] = playlist["pid"]
            result["num_holdouts"] = playlist["num_holdouts"]
            track_vector, artist_vector = recommendations_to_hits(playlist["recommended"], test_dict[playlist["pid"]],
                                                                   track_to_album, album_to_artist)
            result["vector_artist"] = artist_vector
            result["vector_track"] = track_vector
            output_file.write(json.dumps(result) + "\n")
            if i % 20 == 0:
                output_file.flush()
