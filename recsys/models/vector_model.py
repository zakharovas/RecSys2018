import numpy as np
import tensorflow as tf
import random
import utils.serialize_iterable
import utils.load_info_for_model
import implicit
import re
from scipy import sparse


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


class VectorModel:
    def __init__(self, name):
        self._name = name
        self._dict_vectors = dict()

    def __str__(self):
        return self._name

    def load_from_file(self, filename):
        pass
        model_filename = filename
        # with open(model_filename, "rb") as model_file:
        self._dict_vectors = utils.serialize_iterable.load_matrix(model_filename)
        self._dict_vectors = tf.constant(np.array(self._dict_vectors), dtype=tf.float32)

    def load_from_dict(self, matrix):
        self._dict_vectors = tf.constant(matrix, dtype=tf.float32)

    def score_matrix(self, playlist, candidates):
        playlist_matrix = tf.gather(self._dict_vectors, tf.constant(playlist["tracks"], dtype=tf.int32))
        tracks_matrix = tf.gather(self._dict_vectors, tf.constant(candidates, dtype=tf.int32))
        results = tf.matmul(tracks_matrix, playlist_matrix, transpose_b=True)
        return tf.cast(results, tf.float32)

    def get_vector_norm(self, tracks):
        return tf.cast(
            tf.maximum(tf.norm(tf.gather(self._dict_vectors, tf.constant(tracks, dtype=tf.int32)), axis=1), 1e-9),
            tf.float32)


class NumpyVectorModel:
    def __init__(self, name):
        self._name = name
        self._dict_vectors = dict()

    def __str__(self):
        return self._name

    def load_from_file(self, filename):
        pass
        model_filename = filename
        self._dict_vectors = utils.serialize_iterable.load_matrix(model_filename)

    def load_from_dict(self, matrix):
        self._dict_vectors = np.array(matrix)

    def score_matrix(self, playlist, candidates):
        playlist_matrix = self._dict_vectors[playlist["tracks"]]
        tracks_matrix = self._dict_vectors[candidates]
        scores = np.dot(tracks_matrix, playlist_matrix.T)
        return scores

    def get_vector_norm(self, tracks):
        return np.maximum(np.linalg.norm(self._dict_vectors[tracks], axis=1), 1e-9)


class ItemVectorModel(VectorModel):

    def load_from_file(self, track_filename):
        # playlist_matrix = utils.serialize_iterable.load_matrix(playlist_filename)
        track_matrix = utils.serialize_iterable.load_matrix(track_filename)
        self.load_from_dict(track_matrix)

    def load_from_dict(self, track_matrix):
        self._size = len(track_matrix[0])
        # self._playlist_vectors = tf.constant(playlist_matrix, dtype=tf.float32)
        self._dict_vectors = tf.constant(track_matrix, dtype=tf.float32)
        self._model = implicit.als.AlternatingLeastSquares(factors=self._size, num_threads=32,
                                                           calculate_training_loss=True,
                                                           regularization=1)
        self._model.item_factors = track_matrix
        self._implicit = 50

    def _get_playlist_vector(self, playlist):
        tracks = playlist["tracks"]
        ids = np.zeros(len(tracks))
        values = self._implicit * np.ones(len(tracks))
        return self._model.recalculate_user(0, sparse.csr_matrix((values, (ids, tracks)),
                                                                 shape=[1, self._model.item_factors.shape[0]]))

    def score_matrix(self, playlist, candidates):
        playlist_vector = tf.reshape(tf.constant(self._get_playlist_vector(playlist), dtype=tf.float32),
                                     (1, self._size))
        tracks_matrix = tf.gather(self._dict_vectors, tf.constant(candidates, dtype=tf.int32))
        results = tf.matmul(playlist_vector, tracks_matrix, transpose_b=True)

        return tf.cast(results, tf.float32)

    # def multi_score(self, pids, candidates):
    #     playlist_vector = tf.reshape(tf.gather(self._playlist_vectors, tf.constant(pids, dtype=tf.int32)))
    #     tracks_matrix = tf.gather(self._dict_vectors, tf.constant(candidates, dtype=tf.int32))
    #     results = tf.matmul(playlist_vector, tracks_matrix, transpose_b=True)
    #     return results

    def get_playlist_vector_norm(self, playlist):
        return tf.cast(tf.maximum(tf.norm(self._get_playlist_vector(playlist)), 1e-9), tf.float32)


class NumpyItemVectorModel(NumpyVectorModel):
    def load_from_file(self, track_filename):
        track_matrix = utils.serialize_iterable.load_matrix(track_filename)
        self.load_from_dict(track_matrix)

    def load_from_dict(self, track_matrix):
        self._size = len(track_matrix[0])
        self._dict_vectors = track_matrix
        self._model = implicit.als.AlternatingLeastSquares(factors=self._size, num_threads=32,
                                                           calculate_training_loss=True,
                                                           regularization=1)
        self._model.item_factors = track_matrix
        self._implicit = 50

    def _get_playlist_vector(self, playlist):
        tracks = playlist["tracks"]
        ids = np.zeros(len(tracks))
        values = self._implicit * np.ones(len(tracks))
        return self._model.recalculate_user(0, sparse.csr_matrix((values, (ids, tracks)),
                                                                 shape=[1, self._model.item_factors.shape[0]]))

    def score_matrix(self, playlist, candidates):
        playlist_vector = self._get_playlist_vector(playlist).reshape((1, self._size))
        tracks_matrix = self._dict_vectors[candidates]
        results = np.dot(playlist_vector, tracks_matrix.T)
        return results

    def get_playlist_vector_norm(self, playlist):
        return np.max([np.linalg.norm(self._get_playlist_vector(playlist)), 1e-9])


class NameAls(VectorModel):

    def load_from_file(self, playlist_filename, track_filename, name_encoding):
        playlist_matrix = utils.serialize_iterable.load(playlist_filename)
        track_matrix = utils.serialize_iterable.load(track_filename)
        name_encoding = utils.load_info_for_model.load_json(name_encoding)
        self.load_from_dict(playlist_matrix, track_matrix, name_encoding)

    def load_from_dict(self, playlist_matrix, track_matrix, name_encoding):
        self._size = len(playlist_matrix[0])
        self._playlist_vectors = tf.constant(playlist_matrix)
        self._dict_vectors = tf.constant(track_matrix)
        self._name_encoding = name_encoding

    def score_matrix(self, playlist, candidates):
        if "name" not in playlist or normalize_name(playlist["name"]) not in self._name_encoding:
            playlist_vector = np.zeros((self._size))
        else:
            pid = self._name_encoding[normalize_name(playlist["name"])]
            playlist_vector = tf.gather(self._playlist_vectors, tf.constant([pid], dtype=tf.int32))

        playlist_vector = tf.reshape(playlist_vector,
                                     (1, self._size))

        tracks_matrix = tf.gather(self._dict_vectors, tf.constant(candidates, dtype=tf.int32))
        results = tf.matmul(playlist_vector, tracks_matrix, transpose_b=True)

        return results

    def multi_score(self, pids, candidates):
        playlist_vector = tf.reshape(tf.gather(self._playlist_vectors, tf.constant(pids, dtype=tf.int32)))
        tracks_matrix = tf.gather(self._dict_vectors, tf.constant(candidates, dtype=tf.int32))
        results = tf.matmul(playlist_vector, tracks_matrix, transpose_b=True)
        return results

    def get_playlist_vector_norm(self, playlist):
        if "name" not in playlist or normalize_name(playlist["name"]) not in self._name_encoding:
            playlist_vector = np.zeros((self._size))
        else:
            pid = self._name_encoding[normalize_name(playlist["name"])]
            playlist_vector = tf.gather(self._playlist_vectors, tf.constant([pid], dtype=tf.int32))
        return tf.maximum(tf.norm(playlist_vector), 1e-9)


class NumpyNameAls(NumpyVectorModel):

    def load_from_file(self, playlist_filename, track_filename, name_encoding):
        playlist_matrix = utils.serialize_iterable.load(playlist_filename)
        track_matrix = utils.serialize_iterable.load(track_filename)
        name_encoding = utils.load_info_for_model.load_json(name_encoding)
        self.load_from_dict(playlist_matrix, track_matrix, name_encoding)

    def load_from_dict(self, playlist_matrix, track_matrix, name_encoding):
        self._size = len(playlist_matrix[0])
        self._playlist_vectors = playlist_matrix
        self._dict_vectors = track_matrix
        self._name_encoding = name_encoding

    def score_matrix(self, playlist, candidates):
        if "name" not in playlist or normalize_name(playlist["name"]) not in self._name_encoding:
            playlist_vector = np.zeros((self._size))
        else:
            pid = self._name_encoding[normalize_name(playlist["name"])]
            playlist_vector = self._playlist_vectors[pid]
        playlist_vector = playlist_vector.reshape((1, self._size))
        tracks_matrix = self._dict_vectors[candidates]
        results = np.dot(playlist_vector, tracks_matrix.T)

        return results

    def multi_score(self, pids, candidates):
        playlist_vector = tf.reshape(tf.gather(self._playlist_vectors, tf.constant(pids, dtype=tf.int32)))
        tracks_matrix = self._dict_vectors[candidates]
        results = np.dot(playlist_vector, tracks_matrix.T)
        return results

    def get_playlist_vector_norm(self, playlist):
        if "name" not in playlist or normalize_name(playlist["name"]) not in self._name_encoding:
            playlist_vector = np.zeros((self._size))
        else:
            pid = self._name_encoding[normalize_name(playlist["name"])]
            playlist_vector = self._playlist_vectors[pid]
        return np.maximum(np.linalg.norm(playlist_vector), 1e-9)


class SvdPP(VectorModel):

    def load_from_dict(self, name_dict, track_matrix, inner_matrix, item_bias):
        self._size = len(track_matrix[0])
        self._name_dict = name_dict
        self._dict_vectors = tf.constant(track_matrix)
        self._inner_vectors = tf.constant(inner_matrix)
        self._item_bias = tf.constant(item_bias)

    def score_matrix(self, playlist, candidates):
        if "name" not in playlist or normalize_name(playlist["name"]) not in self._name_dict:
            playlist_vector = np.zeros((self._size))
        else:
            playlist_vector = tf.constant(self._name_dict[normalize_name(playlist["name"])])
        if len(playlist["tracks"]) > 0:
            playlist_vector += tf.reduce_sum(
            tf.gather(self._inner_vectors, tf.constant(playlist["tracks"], dtype=tf.int32)), axis=0) / np.sqrt(len(playlist["tracks"]))
        playlist_vector = tf.reshape(playlist_vector, (1, self._size))
        tracks_matrix = tf.gather(self._dict_vectors, tf.constant(candidates, dtype=tf.int32))
        results = tf.matmul(playlist_vector, tracks_matrix, transpose_b=True)
        results += tf.gather(self._item_bias, tf.constant(candidates, dtype=tf.int32))
        return results

    def get_playlist_vector_norm(self, playlist):
        if "name" not in playlist or normalize_name(playlist["name"]) not in self._name_dict:
            playlist_vector = np.zeros((self._size))
        else:
            playlist_vector = tf.constant(self._name_dict[normalize_name(playlist["name"])])
        if len(playlist["tracks"]) > 0:
            playlist_vector += tf.reduce_sum(
            tf.gather(self._inner_vectors, tf.constant(playlist["tracks"], dtype=tf.int32)), axis=0)
        return tf.maximum(tf.norm(playlist_vector), 1e-9)


class NumpySvdPP(NumpyVectorModel):

    def load_from_dict(self, name_dict, track_matrix, inner_matrix, item_bias):
        self._size = len(track_matrix[0])
        self._name_dict = name_dict
        self._dict_vectors = track_matrix
        self._inner_vectors = inner_matrix
        self._item_bias = item_bias

    def score_matrix(self, playlist, candidates):
        if "name" not in playlist or normalize_name(playlist["name"]) not in self._name_dict:
            playlist_vector = np.zeros((self._size))
        else:
            playlist_vector = self._name_dict[normalize_name(playlist["name"])]
        if len(playlist["tracks"]) > 0:
            playlist_vector += np.sum(self._inner_vectors[playlist["tracks"]], 0) / np.sqrt(len(playlist["tracks"]))
        playlist_vector = playlist_vector.reshape((1, self._size))
        tracks_matrix = self._dict_vectors[candidates]
        results = np.dot(playlist_vector, tracks_matrix.T)
        results += self._item_bias[candidates]
        return results

    def get_playlist_vector_norm(self, playlist):
        if "name" not in playlist or normalize_name(playlist["name"]) not in self._name_dict:
            playlist_vector = np.zeros((self._size))
        else:
            playlist_vector = self._name_dict[normalize_name(playlist["name"])]
        if len(playlist["tracks"]) > 0:
            playlist_vector += np.sum(self._inner_vectors[playlist["tracks"]], 0)
        return np.maximum(np.linalg.norm(playlist_vector), 1e-9)
