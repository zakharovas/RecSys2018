import sys
import json
import numpy as np
from tqdm import tqdm
import tensorflow.contrib.eager as tfe
from multiprocessing import Pool
import os

import utils.load_vector_models
import utils.load_info_for_model
from calculate_features.all_features import all_features
from models.vector_model import NumpyVectorModel, NumpyItemVectorModel, NumpyNameAls, NumpySvdPP


def load_popularity(filename):
    raw_popularity = np.array(utils.load_info_for_model.load_json(filename))
    raw_popularity[raw_popularity < 1] = 1
    return raw_popularity


def create_features(playlists):
    result = []
    for playlist in tqdm(playlists):
        if len(playlist["candidates"]) == 0 or len(playlist["tracks"]) == 0:
            continue

        keys, features = all_features(playlist, playlist["candidates"], (vector_models, ui_models, svd_pp, nals) , track_to_album,
                                      album_to_artist, [track_popularity, album_popularity, artist_popularity], with_name, playlist["wv"])
        result.append((keys, features))
    return result


if __name__ == '__main__':
    tfe.enable_eager_execution()
    print(sys.argv)
    playlists_fielname = sys.argv[1]
    track_to_album = sys.argv[2]
    album_to_artist = sys.argv[3]
    vector_model_description = sys.argv[4]
    ui_vector_model_description = sys.argv[5]

    track_popularity_filename = sys.argv[6]
    album_popularity_filename = sys.argv[7]
    artist_popularity_filename = sys.argv[8]
    output_filename = sys.argv[9]
    number_of_threads = int(sys.argv[10])
    with_name = int(sys.argv[11])
    # skip = int(sys.argv[8])
    # lines_to_do = int(sys.argv[9])
    svdpp_folder = "/home/alzaharov/name_vectors/stash"
    nals_folder = "/home/alzaharov/name_vectors"
    track_to_album = np.array(utils.load_info_for_model.load_json(track_to_album))
    album_to_artist = np.array(utils.load_info_for_model.load_json(album_to_artist))

    track_popularity = load_popularity(track_popularity_filename)
    album_popularity = load_popularity(album_popularity_filename)
    artist_popularity = load_popularity(artist_popularity_filename)

    vector_models, ui_models = utils.load_vector_models.load_models_from_info_file(vector_model_description, NumpyVectorModel, NumpyItemVectorModel)
    m, uim, svd_pp = utils.load_vector_models.load_svd_pp_model(svdpp_folder, NumpyVectorModel, NumpyItemVectorModel, NumpySvdPP)
    vector_models.append(m)
    ui_models.append(uim)
    m, uim, nals = utils.load_vector_models.load_name_als(nals_folder, NumpyVectorModel, NumpyItemVectorModel,
                                                                NumpyNameAls)
    vector_models.append(m)
    ui_models.append(uim)

    # ui_models = utils.load_vector_models.load_user_item_models_from_file(ui_vector_model_description, NumpyItemVectorModel)
    # done = 0
    playlists = []
    with open(playlists_fielname, 'r') as input_file:
        playlists = [json.loads(x) for x in input_file]
    # playlists = playlists[-200000:]
    pool = Pool(processes=number_of_threads)
    results = []
    task_size = (len(playlists) + number_of_threads - 1) // number_of_threads // 5
    create_features(playlists[:2])
    for i in range((len(playlists) + task_size - 1) // task_size):
        results.append(pool.apply_async(create_features, (playlists[i * task_size: (i + 1) * task_size],)))
    with open(output_filename, 'w') as output_file:
        print_head = True
        i = 0
        for res in results:
            print ("waiting")
            for result in res.get():
                if print_head:
                    keys = ["target", "groupId"] + result[0]

                    output_file.write('\t'.join(keys) + '\n')
                    print_head = False
                for target, line in zip(playlists[i]["target"], result[1]):
                    output_file.write("{}\t{}\t".format(target, int(playlists[i]["pid"])))
                    output_file.write('\t'.join(map(str, line)) + '\n')
                i += 1
            output_file.flush()
    print("OK")
