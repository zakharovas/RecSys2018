import sys
import utils.serialize_iterable
import numpy as np
import json
from tqdm import tqdm


def load_matrix(filename, size, mode):
    with open(filename) as input_file:
        length = len(input_file.readline().rstrip().split()) - 1
        if mode == 0:
            length += 1
    matrix = np.zeros((size, length))
    with open(filename) as input_file:
        for i, line in enumerate(tqdm(input_file)):

            array = line.rstrip().split()
            if mode == 1:
                id = int(array[0][1:])
                values = np.array([float(x) for x in array[1:]])
                matrix[id] = values
            else:
                values = np.array([float(x) for x in array])
                matrix[i] = values
    return matrix

if __name__ == "__main__":
    vector_filename = sys.argv[1]
    track_to_album_filename = sys.argv[2]
    album_to_artist_filename = sys.argv[3]
    object_type = sys.argv[4]
    output_filename = sys.argv[5]
    mode = int(sys.argv[6])
    with open(track_to_album_filename) as track_to_album_file:
        track_to_album = np.array(json.loads(track_to_album_file.read()))
    with open(album_to_artist_filename) as album_to_artist_file:
        album_to_artist = np.array(json.loads(album_to_artist_file.read()))
    artist_size = np.max(album_to_artist) + 1
    album_size = len(album_to_artist)
    track_size = len(track_to_album)

    if object_type == "track":
        transform_track_id = lambda x: x
        size = track_size
    elif object_type == "album":
        transform_track_id = lambda track_id: track_to_album[track_id]
        size = album_size
    elif object_type == "artist":
        transform_track_id = lambda track_id: album_to_artist[track_to_album[track_id]]
        size = artist_size
    matrix = load_matrix(vector_filename, size, mode)

    output_matrix = np.zeros((track_size, matrix.shape[1]))
    all_track_id = np.arange(0, track_size, 1, dtype=int)
    output_matrix = matrix[transform_track_id(all_track_id)]

    # with open(output_filename, "wb") as  output_file:
    print(output_matrix.shape)
    utils.serialize_iterable.dump_matrix(output_matrix, output_filename)
