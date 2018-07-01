import numpy as np

from calculate_features import counter_features, vector_features, popularity_features, random_walk_features


def all_features(playlist, candidates, models, track_to_album, album_to_artist, popularities, with_name, wv):
    keys = []
    values = []
    hard_candidates = candidates
    candidates = [x[0] for x in candidates[0][2]]
    keys.append("number_tracks")
    holdouts = np.ones((len(candidates), 1))
    if "number_of_tracks" not in playlist:
        holdouts *= len(playlist["tracks"])
    else:
        holdouts *= int(playlist["number_of_tracks"])
    values.append(holdouts)

    counter_keys, counter_features_matrix = counter_features.counter_features(playlist, candidates,
                                                                              track_to_album, album_to_artist)
    keys += counter_keys
    values.append(counter_features_matrix)

    keys.append("wv")
    values.append(np.reshape(np.array(wv), (len(wv), 1)))

    popylarity_keys, popularity_values = popularity_features.popularity_features(playlist, candidates, track_to_album,
                                                                                 album_to_artist, popularities)
    keys += popylarity_keys
    values.append(popularity_values)

    for model in models[0]:
        vector_keys, vector_features_matrix = vector_features.vector_features(playlist, candidates, model)
        keys += vector_keys
        values.append(vector_features_matrix)
    for model in models[1]:
        vector_keys, vector_features_matrix = vector_features.user_item_vector_features(playlist, candidates, model)
        keys += vector_keys
        values.append(vector_features_matrix)
    svdpp_keys, svdpp_matrix = vector_features.user_item_vector_features(playlist, candidates, models[2])
    keys+= svdpp_keys
    values.append(svdpp_matrix)
    if with_name:
        als_name_keys, als_name_matrix = vector_features.user_item_vector_features(playlist, candidates, models[3])
        keys += als_name_keys
        values.append(als_name_matrix)

    keys += ["random"]
    random_vector = np.random.randn(len(candidates), 1)
    values.append(random_vector)
    walk_keys, walk_features = random_walk_features.random_walk_features(playlist, hard_candidates)
    keys += walk_keys
    values.append(walk_features)

    return keys, np.hstack(values)
