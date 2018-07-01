import sys
import json
from tqdm import tqdm


if __name__ == "__main__":
    playlist_filename = sys.argv[1]
    id_filename = sys.argv[2]
    object_type = sys.argv[3]
    output_filename = sys.argv[4]
    with open(id_filename) as id_file:
        id_encoding = json.loads(id_file.read())
    names = [""] * len(id_encoding)
    with open(playlist_filename) as playlist_file:
        for line in tqdm(playlist_file):
            playlist = json.loads(line)
            for track in playlist["tracks"]:
                position = id_encoding[track[object_type+"_uri"]]
                value = track[object_type+"_name"]
                names[position] = value
    with open(output_filename, "w") as output_file:
        output_file.write(json.dumps(names))
