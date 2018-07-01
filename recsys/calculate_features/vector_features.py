import numpy as np
import tensorflow as tf


def vector_features(playlist, candidates, model):
    keys = []
    percentile = [0, 50, 75, 90, 95, 100]
    # norms = tf.norm()
    for x in percentile:
        keys.append("{}_dot_perc_{}".format(str(model), x))
    values = np.zeros((4 * (len(percentile) + 1), len(candidates)))
    scores_matrix = model.score_matrix(playlist, candidates)
    playlist_vectors = tf.maximum(model.get_vector_norm(playlist["tracks"]), 1e-9)
    candidates_vectros = tf.maximum(tf.reshape(model.get_vector_norm(candidates), (len(candidates), 1)), 1e-9)
    for i, x in enumerate(percentile):
        values[i] = tf.contrib.distributions.percentile(scores_matrix, x, axis=1)

    keys.append("{}_dot_average".format(str(model)))
    values[len(percentile)] = tf.reduce_mean(scores_matrix, axis=1).numpy()
    # cos, / cand, / pl
    transforms = [lambda matrix: (matrix / playlist_vectors) / candidates_vectros,
                  lambda matrix: matrix / candidates_vectros,
                  lambda matrix: matrix / playlist_vectors]
    names = ["cos", "cos_with_playlist_norm", "cos_with_candidate_norm"]
    for j, (name, transform) in enumerate(zip(names, transforms)):
        current_matrix = transform(scores_matrix)
        for x in percentile:
            keys.append("{}_{}_perc_{}".format(str(model), name, x))
        keys.append("{}_{}_average".format(str(model), name))
        for i, x in enumerate(percentile):
            values[(1 + j) * (len(percentile) + 1) + i] = tf.contrib.distributions.percentile(current_matrix, x, axis=1)
        values[(1 + j) * (len(percentile) + 1) + len(percentile)] = tf.reduce_mean(current_matrix, axis=1).numpy()
    return keys, values.T

def user_item_vector_features(playlist, candidates, model):
    keys = []
    values = []
    keys.append("{}_dot".format(str(model)))
    scores = model.score_matrix(playlist, candidates)[0]
    values.append(scores)
    candidates_norm = model.get_vector_norm(candidates)
    playlist_norm = model.get_playlist_vector_norm(playlist)
    transforms = [lambda matrix: (matrix / playlist_norm) / candidates_norm,
                  lambda matrix: matrix / candidates_norm,
                  lambda matrix: matrix / playlist_norm,
                  lambda matrix: candidates_norm]
    names = ["cos", "cos_with_playlist_norm", "cos_with_candidate_norm", "candidates_norm"]
    for name, transform in zip(names, transforms):
        values.append(transform(scores))
        keys.append("{}_{}".format(str(model), name))
    values = np.vstack(values)
    return keys, values.T


def all_vector_features(playlist, candidates, models):
    keys = []
    values = []
    for model in models:
        vector_keys, vector_features_matrix = vector_features(playlist, candidates, model)
        keys += vector_keys
        values.append(vector_features_matrix)
    values = np.hstack(values)
    return keys, values
