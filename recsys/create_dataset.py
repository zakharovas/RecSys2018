import json
import numpy as np
import sys
from tqdm import tqdm

import utils.load_info_for_model
import candidate_generator.random_walk

if __name__ == '__main__':
    number_of_candidate_negatives = int(sys.argv[2])
    number_of_random_negatives = int(sys.argv[3])
    todo = int(sys.argv[5])
    skip = int(sys.argv[6])
    with open(sys.argv[1], 'r') as train, open(sys.argv[4], 'w') as output:
        train = json.loads(train.read())
        all_tracks = set()
        for playlist in tqdm(train):
            for track in playlist["tracks"]:
                all_tracks.add(track)
        all_tracks = list(all_tracks)
        random_negatives = np.random.choice(all_tracks, size=len(train) * number_of_random_negatives)
        candidate_generator = candidate_generator.random_walk.RandomWalkCandidates(0.7, train,
                                                                                   number_of_candidate_negatives, 30)
        tracks_to_delete = np.random.randint(0, 1000, len(train))
        for i, playlist in tqdm(list(enumerate(train))):
            if i < skip:
                continue
            if i >= skip + todo:
                break
            size = len(playlist["tracks"])
            tracks_to_delete = int(tracks_to_delete[i] % size)
            tracks = []
            deleted_track = playlist["tracks"][track_to_delete]
            del playlist["tracks"][track_to_delete]

            pid = playlist["pid"]
            for x in list(playlist):
                if x != "pid" and x != "tracks":
                    del playlist[x]
            negatives = candidate_generator.create_candidates(playlist)
            current_random_negatives = list(
                random_negatives[i * number_of_random_negatives: (i + 1) * number_of_random_negatives])
            negatives = negatives + current_random_negatives
            if track_to_delete in negatives:
                negatives.remove(track_to_delete)
            playlist["position"] = track_to_delete
            playlist["deleted_track"] = [deleted_track] + negatives
            playlist["target"] = [1] + [0] * (len(negatives))
            output.write(json.dumps(playlist) + "\n")
