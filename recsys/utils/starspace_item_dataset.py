import sys
import json
from tqdm import tqdm

from training import calculate_popularity

if __name__ == '__main__':
    print(sys.argv)
    train_filename = sys.argv[1]
    track_to_album_filename = sys.argv[2]
    album_to_artist_filename = sys.argv[3]
    object_type = sys.argv[4]
    output_filename = sys.argv[5]
    mode = int(sys.argv[6])
    with open(track_to_album_filename) as track_to_album_file:
        track_to_album = json.loads(track_to_album_file.read())
    with open(album_to_artist_filename) as album_to_artist_file:
        album_to_artist = json.loads(album_to_artist_file.read())
    if object_type == "track":
        get_function = calculate_popularity.get_tracks
    elif object_type == "album":
        get_function = lambda playlist: calculate_popularity.get_albums(playlist, track_to_album)
    elif object_type == "artist":
        get_function = lambda playlist: calculate_popularity.get_artist(playlist, album_to_artist, track_to_album)

    with open(train_filename) as train_file, open(output_filename, "w") as output_file:
        for line in tqdm(train_file):
            playlist = json.loads(line)
            objects = set(get_function(playlist))
            if len(objects) == 0:
                continue
            if mode == 1:
                output_file.write("{}\t".format(playlist["pid"]))
                output_file.write('\t'.join(str(x) for x in objects) + "\n")
            else:
                output_file.write('\t'.join("l" + str(x) for x in objects) + "\n")
