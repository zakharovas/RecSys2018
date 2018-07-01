import numpy as np
import tensorflow as tf

def popularity_features(playlist, candidates, track_to_album, album_to_artist, popularities):
    keys = []
    values = []
    # samo  - /
    candidate_transormation = [lambda x: x, lambda x: track_to_album[x], lambda x: album_to_artist[x]]
    names = ["track", "album", "artist"]
    playlist = playlist["tracks"]
    for popularity_vector, transormation, name in zip(popularities, candidate_transormation, names):
        keys.append("{}_candidate_popularity".format(name))
        candidates = transormation(candidates)
        playlist = transormation(playlist)
        candidate_popularity = popularity_vector[candidates]
        playlist_popularity = popularity_vector[playlist]
        values.append(candidate_popularity)
        # values = np.vstack((values, candidate_popularity))
        percentile = [50, 75, 90, 95, 100]
        playlist_values = np.percentile(playlist_popularity, percentile)
        playlist_as_matrix = np.transpose(np.reshape(np.tile(playlist_values, [len(candidates)]), [len(candidates), len(playlist_values)]))
        values.append(playlist_as_matrix - candidate_popularity)
        values.append(playlist_as_matrix / candidate_popularity)
        # values = np.vstack((values, playlist_as_matrix - candidate_popularity))
        # values = np.vstack((values, playlist_as_matrix / candidate_popularity))
        for x in percentile:
            keys.append("{}_{}_popularity_playlist_diff".format(name, x))
        for x in percentile:
            keys.append("{}_{}_popularity_playlist_fraction".format(name, x))
        average_vector = np.repeat(np.average(playlist_values), len(candidates))
        keys.append("{}_average_playlist_popularity".format(name))
        keys.append("{}_average_playlist_popularity_diff".format(name))
        keys.append("{}_average_playlist_popularity_fraction".format(name))
        average_matrix = np.zeros((3, len(candidates)))
        average_matrix[0] = average_vector
        average_matrix[1] = average_vector - candidate_popularity
        average_matrix[2] = average_vector / candidate_popularity

        values.append(average_matrix)
    values = np.vstack(values)
    return keys, values.T
