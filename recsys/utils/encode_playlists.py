import sys
import json
from tqdm import tqdm

def get_playlists(filename):
    with open(filename) as current_file:
        content = json.loads(current_file.read())
        return content["playlists"]

if __name__ == '__main__':
    FIELDS = ["tracks", "deleted_track"]
    playlist_filename =sys.argv[1]
    track_filename = sys.argv[2]
    encoded_playlists_filename = sys.argv[3]
    with open(track_filename) as track_file:
        track_ids = json.loads(track_file.read())
    with open(encoded_playlists_filename, "w") as output_file:
        for line in tqdm(get_playlists(playlist_filename)):
            playlist = line
            for field in FIELDS:
                if field in playlist:
                    new_field = []
                    for item in playlist[field]:
                        if isinstance(item, str):
                            new_field.append(track_ids[item])
                        else:
                            new_field.append(track_ids[item["track_uri"]])
                    playlist[field] = new_field
            output_file.write(json.dumps(playlist) + "\n")
