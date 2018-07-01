import sys
import json
from tqdm import tqdm

if __name__ == "__main__":
    full_playlists = dict()
    partial_filename = sys.argv[1]
    full_test_filename = sys.argv[2]
    output_filename = sys.argv[3]
    with open(full_test_filename) as full_test_file:
        for line in tqdm(full_test_file):
            playlist = json.loads(line)
            full_playlists[playlist["pid"]] = playlist
    with open(partial_filename) as partial_file, open(output_filename, "w") as output_file:
        for line in tqdm(partial_file):
            current_playlist = json.loads(line)
            all_tracks = full_playlists[current_playlist["pid"]]["tracks"]
            deleted = set(current_playlist["deleted_track"])
            for track in all_tracks:
                if track not in deleted:
                    current_playlist["tracks"].append(track)
            output_file.write(json.dumps(current_playlist) + "\n")
