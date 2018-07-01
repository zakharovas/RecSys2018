import re
from json import loads
from json import dumps

from numpy import save
from scipy import sparse

import sys

import implicit


def normalize_name(name):
    """
    Copy-paste from random_walk file
    """
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def dump_matrix(matrix, filename):
    """
    Copy-paste from utils.serialize
    """
    with open(filename, "wb") as output_file:
        save(output_file, matrix)

def playlist_names_to_tracks(train_filename):
    """
    Make the graph from playlist names to tracks
    as a dictionary
    """
    with open(train_filename, 'r') as train:
        playlist_names_encoding = {}
        playlist_names_to_tracks = {}
        for line in train:
            playlist = loads(line)
            playlist_name = normalize_name(playlist['name'])
            if not playlist_name in playlist_names_to_tracks:
                playlist_names_encoding[playlist_name] = len(playlist_names_to_tracks)
                playlist_names_to_tracks[playlist_name] = []

            playlist_names_to_tracks[playlist_name] += playlist['tracks']

    return playlist_names_encoding, playlist_names_to_tracks

def coo_adjacency_matrix(encoding, pnames_to_tracks, value=1.):
    """
    Return adjacency sparse.coo_matrix from dictionary and
    playlist encoding names
    """
    entries = []
    p_index = []
    t_index = []
    for playlist_name in pnames_to_tracks:
        for tid in pnames_to_tracks[playlist_name]:
            t_index.append(tid)
            p_index.append(encoding[playlist_name])
            entries.append(value)

    matrix = sparse.coo_matrix(
        (entries, (t_index, p_index)),
        shape=[max(t_index) + 1, len(encoding)]
    )
    return matrix.tocsr()

def als_train_and_dump(train_filename,
                       user_filename,
                       item_filename,
                       playlist_names_encoding_filename='encoding.json',
                       factors=256,
                       num_threads=32,
                       regularization=0.01):
    """
    Train the ALS model and dump the user-item vectors
    to corresponding files. The encoding of names is dumped to
    the 'playlist_names_encoding_filename'.

    Here the 'items' are tracks and the 'users' are playlists
    """
    encoding, pnames_to_tracks = playlist_names_to_tracks(train_filename)
    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 num_threads=num_threads,
                                                 calculate_training_loss=True,
                                                 regularization=regularization)
    model.fit(coo_adjacency_matrix(encoding, pnames_to_tracks))
    with open(playlist_names_encoding_filename, 'w') as output_file:
        output_file.write(dumps(encoding))
    dump_matrix(model.item_factors, item_filename)
    dump_matrix(model.user_factors, user_filename)
    return encoding, model.item_factors, model.user_factors

def dump_predictions(train_filename):
    encoding, track_vectors, name_vectors = als_train_and_dump(
        train_filename,
        'utest.npy',
        'itest.npy',
        factors=256
    )

dump_predictions(sys.argv[1])
