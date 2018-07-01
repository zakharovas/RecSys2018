import json
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing as mp

import candidate_generator.random_walk
import candidate_generator.complex_generator
import utils.serialize_iterable


def create_candidates(playlists):
    candidates = []
    for iteration, playlist_ in enumerate(tqdm(playlists)):
        current_candidate = []
        # no-name#
        if len(playlist_["tracks"]) > 0:
            if "name" not in playlist_ or len(playlist_["name"]) == 0 or playlist_["name"] == "#no-name#" or len(
                    playlist["tracks"]) > 20 or ("number_of_tracks" in playlist and playlist["number_of_tracks"] > 20):
                current_candidate = no_name_generator.candidates(playlist_)
            else:
                current_candidate = name_comp_generator.candidates(playlist_)
        else:
            current_candidate = name_generator.walk(playlist_)
            current_candidate = (current_candidate[0], current_candidate[1][:1000])
        candidates.append(current_candidate)
    return candidates


if __name__ == '__main__':
    number_of_threads = int(sys.argv[4])
    test_playlists = []
    test_filename = sys.argv[1]
    train_filename = sys.argv[2]
    candidate_filename = sys.argv[3]
    train_playlists = []
    with open(train_filename, 'r') as train:
        for i, line in tqdm(enumerate(train)):
            playlist = json.loads(line)
            # for label in list(playlist):
            #     if label != "pid" and label != "tracks" and label != "name":
            #         del playlist[label]
            train_playlists.append(playlist)
    print("TRAIN READ")
    with open(test_filename) as test_file:
        test_playlists = [json.loads(line) for line in test_file]
    print(test_playlists[0])
    print(len(test_playlists))
    max_track = max([len(x["tracks"]) for x in test_playlists])
    min_track = min([len(x["tracks"]) for x in test_playlists])
    print(min_track, max_track)
    name_generator = candidate_generator.random_walk.NameRandomWalk(0.7, train_playlists,
                                                                    10000, 30)
    generator = candidate_generator.random_walk.RandomWalkCandidates(0.7, train_playlists,
                                                                     10000, 30)
    no_name_generator = candidate_generator.complex_generator.ComplexGenerator([generator], 1000)
    name_comp_generator = candidate_generator.complex_generator.ComplexGenerator([generator, name_generator], 1000)
    print(create_candidates(test_playlists[:2])[0])
    pool = mp.Pool(processes=number_of_threads)

    results = []
    task_size = ((len(test_playlists) + number_of_threads - 1) // number_of_threads + 2) // 3

    for i in range((len(test_playlists) + task_size - 1) // task_size):
        results.append(pool.apply_async(create_candidates, (test_playlists[i * task_size: (i + 1) * task_size],)))
    full_results = []
    for result in results:
        full_results += result.get()
    utils.serialize_iterable.dump(full_results, candidate_filename)
