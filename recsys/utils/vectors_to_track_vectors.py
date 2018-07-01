import utils.serialize_iterable
import sys
import json
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    model_filename = sys.argv[1]
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
        transform = lambda id: id
    elif object_type == "album":
        transform = lambda id: track_to_album[id]
    elif object_type == "artist":
        transform = lambda id: album_to_artist[track_to_album[id]]
    with open(model_filename, "rb") as input_file:
        current_matrix = np.array(utils.serialize_iterable.load_matrix(model_filename))
    matrix = np.zeros((track_size, current_matrix.shape[1]))
    for i in range(track_size):
        matrix[i] = current_matrix[transform(i)]
    utils.serialize_iterable.dump_matrix(matrix, output_filename)
