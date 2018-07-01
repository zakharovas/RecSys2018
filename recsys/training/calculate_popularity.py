import json
import sys
import numpy as np
from tqdm import tqdm


def get_tracks(playlist):
    return playlist["tracks"]


def get_albums(playlist, track_to_album):
    return [track_to_album[x] for x in get_tracks(playlist)]


def get_artist(playlist, album_to_artist, track_to_album):
    return [album_to_artist[x] for x in get_albums(playlist, track_to_album)]


if __name__ == "__main__":
    train_filename = sys.argv[1]
    track_to_album_filename = sys.argv[2]
    album_to_artist_filename = sys.argv[3]
    object_type = sys.argv[4]
    output_filename = sys.argv[5]
    with open(track_to_album_filename) as track_to_album_file:
        track_to_album = json.loads(track_to_album_file.read())
    with open(album_to_artist_filename) as album_to_artist_file:
        album_to_artist = json.loads(album_to_artist_file.read())
    artist_size = np.max(album_to_artist) + 1
    album_size = len(album_to_artist)
    track_size = len(track_to_album)

    if object_type == "track":
        get_function = get_tracks
        size = track_size
    elif object_type == "album":
        get_function = lambda playlist: get_albums(playlist, track_to_album)
        size = album_size
    elif object_type == "artist":
        get_function = lambda playlist: get_artist(playlist, album_to_artist, track_to_album)
        size = artist_size
    stats = np.zeros(size)
    with open(train_filename) as train_file:
        for line in tqdm(train_file):
            objects = get_function(json.loads(line))
            for object in objects:
                stats[object] += 1

    with open(output_filename, "w") as output_file:
        output_file.write(json.dumps(list(stats)))
