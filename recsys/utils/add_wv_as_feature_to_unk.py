import sys
import json

if __name__ == "__main__":
    train_file = sys.argv[1]
    candidate_file = sys.argv[2]
    feature = sys.argv[3]
    output = sys.argv[4]
    with open(train_file, 'r') as input_file:
        playlists = [json.loads(x) for x in input_file]
    with open(candidate_file, 'r') as input_file:
        candidates = [json.loads(x) for x in input_file]

    numbers = []
    with open(feature) as feature_file:
        for line in feature_file:
            numbers.append(float(line))
    i = 0
    with open(output, "w") as output_file:
        for playlist, candidate in zip(playlists, candidates):
            if len(playlist["tracks"]) > 0:
                deleted_tracks = [x[0] for x in candidate[0][2]]
                wv = []
                for _ in deleted_tracks:
                    wv.append(numbers[i])
                    i+=1
                playlist["wv"] = wv
            output_file.write(json.dumps(playlist) + "\n")
