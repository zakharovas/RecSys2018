import sys
import json
from tqdm import tqdm

if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    number_of_tracks = int(sys.argv[3])
    with open(input_filename) as input_file:
        with open(output_filename, "w") as output_file:
            for line in tqdm(input_file):
                playlist = json.loads(line)
                playlist["number_of_tracks"] = number_of_tracks
                output_file.write(json.dumps(playlist) + "\n")
