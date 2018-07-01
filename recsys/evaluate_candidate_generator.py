import sys
import json

import numpy as np
from tqdm import tqdm

import utils.load_info_for_model
from candidate_generator import random_walk


def evaluate(playlists, candidate_generator):
    sizes = [10, 100, 500, 1000, 5000, 10000, 20000, 30000]
    metrics = [[], [], [], [], [], [], [], []]
    for i, playlist in enumerate(tqdm(playlists[:500])):
        candidates = candidate_generator.create_candidates(playlist)
        # print(candidates)
        tracks = set(playlist["deleted_track"])
        holdouts = len(playlist["deleted_track"])
        candidates = [int(x in tracks) for x in candidates]
        # print(len(candidates))
        candidates = np.cumsum(candidates)
        # print(len(candidates))

        for metric, size in zip(metrics, sizes):
            if size > len(candidates):
                metric.append(0)
                continue
            value = candidates[size - 1] / np.min([holdouts, size])
            metric.append(value)
    with open(sys.argv[3], "w") as output_file:
        output_file.write("{}\t{}\t{:.1f}\n".format(candidate_generator._candidate_size, prob, visits))
        for metric, size in zip(metrics, sizes):
            output_file.write("{}\t{:.8f}\n".format(size, np.average(metric)))

if __name__ == '__main__':
    playlists = utils.load_info_for_model.load_lines_json(sys.argv[1])
    print("TEST READ")
    output = []
    with open(sys.argv[2], "r") as file:
        for line in tqdm(file):
            output.append(json.loads(line))
    train = output
    print("TRAIN READ")
    size = int(sys.argv[4])
    prob = float(sys.argv[5])
    visits = int(sys.argv[6])
    print(size, prob, visits)
    candidate_generator = random_walk.RandomWalkCandidates(prob, train, size, visits)
    evaluate(playlists, candidate_generator)
