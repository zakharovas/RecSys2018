import sys
import json

if __name__ == "__main__":
    train_file = sys.argv[1]
    feature = sys.argv[2]
    output = sys.argv[3]
    with open(train_file, 'r') as input_file:
        playlists = [json.loads(x) for x in input_file]
    numbers = []
    with open(feature) as feature_file:
        for line in feature_file:
            numbers.append(float(line))
    i = 0
    with open(output, "w") as output_file:
        for playlist in playlists:
            deleted_tracks = [x[0] for x in playlist["candidates"][0][2]]
            wv = []
            for _ in deleted_tracks:
                wv.append(numbers[i])
                i+=1
            playlist["wv"] = wv
            output_file.write(json.dumps(playlist) + "\n")
