import numpy as np


def counter_features(playlist, candidates, track_to_album, album_to_artist):
    keys = []
    values = []

    album_size = album_to_artist.size
    artist_size = np.max(album_to_artist) + 1
    album_playlist = track_to_album[playlist["tracks"]]
    album_candidates = track_to_album[candidates]

    x = np.zeros(album_size)
    count = np.bincount(album_playlist)
    x[:count.size] = count
    album_intersection = x[album_candidates] / len(playlist["tracks"])
    keys.append("album_intersection")
    values.append(album_intersection)

    x = np.zeros(artist_size)
    count = np.bincount(album_to_artist[album_playlist])
    x[:count.size] = count
    artist_intersection = x[album_to_artist[album_candidates]] / len(playlist["tracks"])
    keys.append("artist_intersection")
    values.append(artist_intersection)

    return keys, np.array(values).T
