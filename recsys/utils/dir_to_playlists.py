import sys
import json
import os
from tqdm import tqdm

TEST = [0, 100, 200, 300, 400, 500, 600, 700, 800, 999]


def get_playlists(filename):
    with open(filename) as current_file:
        content = json.loads(current_file.read())
        return content["playlists"]


def print_playlist(playlists, file):
    for playlist in playlists:
        file.write(json.dumps(playlist) + "\n")


if __name__ == '__main__':
    data_dir = sys.argv[1]
    train_filename = sys.argv[2]
    test_filename = sys.argv[3]
    all_filename = sys.argv[4]
    with open(all_filename, "w") as all_file, open(test_filename, "w") as test_file, \
            open(train_filename, "w") as train_file:
        files = sorted(os.listdir(data_dir))
        for i in tqdm(range(1000)):
            filename = "mpd.slice.{}-{}.json".format(i * 1000, i * 1000 + 999)
            filename = os.path.join(data_dir, filename)
            playlists = get_playlists(filename)
            print_playlist(playlists, all_file)
            if i in TEST:
                print_playlist(playlists, test_file)
            else:
                print_playlist(playlists, train_file)
