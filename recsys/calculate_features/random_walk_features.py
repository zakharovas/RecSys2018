import numpy as np

def random_walk_features(playlist, candidates):
    keys = ["_rank", "_score", "_normalized_max_score", "_normalized_sum_score"]
    output_keys = []
    features = []
    for candidate_score in candidates:
        name = candidate_score[0]
        ranks = np.array([x[2] for x in candidate_score[2]], dtype=np.float32)
        scores = np.array([x[1] for x in candidate_score[2]], dtype=np.float32)
        for key in keys:
            output_keys.append(name + key)
        features.append(ranks)
        features.append(scores)
        features.append(scores / candidate_score[1])
        features.append(scores / np.max([np.max(scores), 1e-9]))
    features = np.vstack(features)
    return output_keys, features.T

