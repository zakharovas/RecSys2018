import implicit
import json
import sys
from scipy import sparse
import numpy as np
from tqdm import tqdm
import pickle
import utils.serialize_iterable

def read_input(filenanme, size):
    data = []
    i = []
    j = []
    with open(filenanme) as input_file:
        for line in tqdm(input_file):
            pid = int(line.rstrip().split()[0])
            for number in line.rstrip().split()[1:]:
                i.append(int(number))
                j.append(pid + 1)
                data.append(value)
    matrix = sparse.coo_matrix((data, (i, j)), shape=[size, max(j) + 1])
    return matrix.tocsr()

def train_als(matrix):
    model = implicit.als.AlternatingLeastSquares(factors=dim, num_threads=32, calculate_training_loss=True,
                                                 regularization=reg)
    model.fit(matrix,)
    return model


def save_vectors(model, output_filename):
    utils.serialize_iterable.dump_matrix(model.item_factors, output_filename)
    utils.serialize_iterable.dump_matrix(model.user_factors, user_filename)
    # # with open(output_filename, "wb") as  output_file:
    # #     print(model.item_factors.shape)
    # #     pickle.dump(model.item_factors.tolist(), output_file)
    # with open(user_filename, "wb") as user_file:
    #     pickle.dump(model.user_factors.tolist(), user_file)


if __name__ == "__main__":
    train_filename = sys.argv[1]
    track_to_album_filename = sys.argv[2]
    album_to_artist_filename = sys.argv[3]
    object_type = sys.argv[4]
    output_filename = sys.argv[5]
    user_filename = sys.argv[6]
    dim = int(sys.argv[7])
    value = float(sys.argv[8])
    reg = float(sys.argv[9])
    with open(track_to_album_filename) as track_to_album_file:
        track_to_album = json.loads(track_to_album_file.read())
    with open(album_to_artist_filename) as album_to_artist_file:
        album_to_artist = json.loads(album_to_artist_file.read())
    artist_size = np.max(album_to_artist) + 1
    album_size = len(album_to_artist)
    track_size = len(track_to_album)
    if object_type == "track":
        size = track_size
    elif object_type == "album":
        size = album_size
    elif object_type == "artist":
        size = artist_size
    matrix = read_input(train_filename, size)
    model = train_als(matrix)
    save_vectors(model, output_filename)
