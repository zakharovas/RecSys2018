import numpy as np
import sys
import json
from tqdm import tqdm


def song_clicks(matrix, holdouts):
    length = matrix.shape[1]
    first_one = np.argmax(matrix, axis=1)
    metric = np.floor(first_one / 10)
    not_found = matrix[np.arange(matrix.shape[0]), first_one] == 0
    max_metric = np.ceil(length / 10) + 1
    metric[not_found] = max_metric
    return np.mean(metric)


def recall(matrix, holdout):
    length = matrix.shape[1]
    holdout = holdout[:]
    holdout[holdout > length] = length
    holdout[holdout == 0] = 1
    extra_retrieved = np.cumsum(matrix, axis=1)
    extra_retrieved = 1 - (extra_retrieved.T > holdout).T
    playlist_precision = np.sum(matrix * extra_retrieved, axis=1) / holdout
    return np.mean(playlist_precision)


def ndcg(matrix, holdout):
    length = matrix.shape[1]
    holdout = holdout[:]
    holdout[holdout > length] = length
    holdout[holdout == 0] = 1
    weights = 1 / np.log2(np.append([2], np.arange(2, length + 1)))
    idcg = np.cumsum(weights)
    extra_retrieved = np.cumsum(matrix, axis=1)
    extra_retrieved = 1 - (extra_retrieved.T > holdout).T
    dcg = np.sum(matrix * extra_retrieved * weights, axis=1)
    ndcg_on_playlist = (dcg.T / idcg[holdout - 1]).T
    return np.mean(ndcg_on_playlist)


def rprec(matrix, holdout):
    length = matrix.shape[1]
    holdout = holdout[:]
    holdout[holdout > length] = length
    cumulated_results = np.cumsum(matrix, 1)
    # print(cumulated_results)
    # print(cumulated_results[np.arange(len(holdout)), holdout - 1] )
    # print(holdout)
    # cumulated_results
    holdout[holdout == 0] = 1

    return np.average(cumulated_results[np.arange(len(holdout)), holdout - 1] / holdout)

def contest_metrics(data):
    holdouts = []
    artist_matrix = []
    track_matrix = []
    lengths = []
    for item in data:
        holdouts.append(item["num_holdouts"])
        artist_matrix.append(item["vector_artist"])
        track_matrix.append(item["vector_track"])
        lengths.append(len(item["vector_track"]))
    holdouts = np.array(holdouts)
    length = np.max(lengths)
    for line in artist_matrix:
        if len(line) < length:
            line += [0] * (length - len(line))
    for line in track_matrix:
        if len(line) < length:
            line += [0] * (length - len(line))
    artist_matrix = np.array(artist_matrix)
    track_matrix = np.array(track_matrix)
    matrixes = (artist_matrix, track_matrix)
    prediction_size = [10, 50, 100, 500, 1000, 5000, 10000, 1000 * 1000 * 1000]
    stats = []
    keys = []
    for i, size in enumerate(prediction_size):
        if i == 0:
            keys.append("NUMBER OF RECOMMENDED")
        current_stat = {"NUMBER OF RECOMMENDED": size}
        for matrix, data_name in zip(matrixes, ["artist", "track"]):
            for metric, metric_name in zip([recall, ndcg, song_clicks, rprec], ["recall", "NDCG", "song_clicks", "Rprec"]):
                if i == 0:
                    keys.append("{}_{}".format(data_name, metric_name))
                current_stat["{}_{}".format(data_name, metric_name)] = metric(matrix[:, :size], np.copy(holdouts))
        stats.append(current_stat)
    return keys, stats


def read_predictions(filename):
    with open(filename) as input_file:
        return [json.loads(line) for line in tqdm(input_file)]


def print_table(keys, stats):
    lines = [[]]
    for x in keys:
        lines[0].append(x)
    for stat in stats:
        line = []
        for x in keys:
            if isinstance(stat[x], float):
                line.append("{:.8f}".format(stat[x]))
            else:
                line.append(str(stat[x]))
        lines.append(line)
    lines = ["|| " + " | ".join(x) + " ||" for x in lines]
    print('\n'.join(lines))


if __name__ == '__main__':
    data = read_predictions(sys.argv[1])
    keys, stats = contest_metrics(data)
    print_table(keys, stats)
