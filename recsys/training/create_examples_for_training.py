import json
import numpy as np
import sys
from tqdm import tqdm
import multiprocessing as mp

import candidate_generator.random_walk
import candidate_generator.complex_generator


def get_sort_permutation(result):

    return np.argsort([-x[1] for x in result])

def drop_candidates(candidates, target):
    length_of_candidates = len(candidates[0][2])
    positions = [i for i, x in enumerate(target) if x == 1]
    uniform_positions = np.random.randint(0, length_of_candidates, uniform_size, np.int32)
    positions += uniform_positions.tolist()
    p = 1 / 1000
    probabilities = (1 - p) ** np.arange(length_of_candidates) * p
    probabilities = probabilities / np.sum(probabilities)
    for generation_result in candidates:
        sorted_permutation = get_sort_permutation(generation_result[2])
        random = np.random.choice(length_of_candidates, geometric_size, replace=False, p=probabilities).tolist()
        positions += [sorted_permutation[x] for x in random]
    positions = set(positions)
    return positions


def create_candidates(train):
    result = []
    for playlist_ in tqdm(train):
        if with_name and "name" not in playlist_ or len(playlist_["name"]) == 0:
            continue
        tracks = set(playlist_["tracks"])
        if len(tracks) == 0:
            continue
        if (len(playlist_["tracks"]) - 5) <= expected_size:
            continue
        remain_size = expected_size
        positions = set(np.random.permutation(np.arange(len(playlist_["tracks"])))[:remain_size])
        remained_tracks = [x for j, x in enumerate(playlist_["tracks"]) if j in positions]
        playlist_["tracks"] = remained_tracks
        candidates = generator.candidates(playlist_)
        target = [int(x[0] in tracks) for x in candidates[0][2]]
        positions = drop_candidates(candidates, target)
        # candidates = generator.walk(playlist_)
        new_candidates = []
        for generation_result in candidates:
            new_candidates.append([generation_result[0], generation_result[1], []])
        for position in positions:
            for i, generation_result in enumerate(candidates):
                new_candidates[i][2].append(generation_result[2][position])

        playlist_["candidates"] = new_candidates
        playlist_["target"] = [target[i] for i in positions]
        result.append(playlist_)
    return result


if __name__ == '__main__':
    candidate_file = sys.argv[1]
    train_file = sys.argv[2]
    uniform_size = int(sys.argv[3])
    geometric_size = int(sys.argv[4])
    with_name = int(sys.argv[5])
    expected_size = int(sys.argv[6])
    number_of_threads = int(sys.argv[7])
    output_filename = sys.argv[8]
    train_playlists = []

    with open(candidate_file, 'r') as train:
        for i, line in tqdm(enumerate(train)):
            playlist = json.loads(line)
            for label in list(playlist):
                if label != "pid" and label != "tracks" and label != "name":
                    del playlist[label]
            train_playlists.append(playlist)
    np.random.seed(42)
    if with_name:
        train_playlists = np.random.permutation(train_playlists).tolist()
    else:
        train_playlists = np.random.permutation(train_playlists).tolist()[50000:]
    with open(train_file) as all_train:
        generator_playlists = [json.loads(line) for line in tqdm(all_train)]
    generator = candidate_generator.random_walk.RandomWalkCandidates(0.7, generator_playlists,
                                                                     10000, 30)
    if with_name == 1:
        name_generator = candidate_generator.random_walk.NameRandomWalk(0.7, generator_playlists,
                                                                         10000, 30)
        generator = candidate_generator.complex_generator.ComplexGenerator([generator, name_generator], 10000)
    else:
        generator = candidate_generator.complex_generator.ComplexGenerator([generator], 10000)
    print(train_playlists[0])
    max_track = max(len(x["tracks"]) for x in train_playlists)
    pool = mp.Pool(processes=number_of_threads)

    results = []
    task_size = 250
    for i in range((len(train_playlists) + task_size - 1) // task_size):
        results.append(pool.apply_async(create_candidates, (train_playlists[i * task_size: (i + 1) * task_size],)))
    with open(output_filename, "w") as output:
        for res in results:
            for line in res.get():
                output.write(json.dumps(line) + "\n")
