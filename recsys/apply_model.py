import sys
import json
import numpy as np
from tqdm import tqdm
import tensorflow.contrib.eager as tfe
# import pickle

import utils.load_info_for_model
import utils.serialize_iterable
import calculate_features.all_features
from models.catboost_model import CatboostModel
import recommendations_to_hits
import utils.load_vector_models
import utils.load_info_for_model
from  models.vector_model import VectorModel, ItemVectorModel, NumpySvdPP, NumpyNameAls, NameAls


def load_popularity(filename):
    raw_popularity = np.array(utils.load_info_for_model.load_json(filename))
    raw_popularity[raw_popularity < 1] = 1
    return raw_popularity


def apply(playlists, candidates, output_filename):
    with open(output_filename, "w") as output_file:
        for i, (playlist, candidate) in enumerate(tqdm(zip(playlists, candidates))):
            result = dict()
            result["pid"] = playlist["pid"]
            # print(playlist)
            # candidate[1] = candidate[1][:1000]
            # playlist["candidate"] = candidate
            if "num_holdouts" in playlist:
                result["num_holdouts"] = playlist["num_holdouts"]
            else:
                result["num_holdouts"] = len(playlist["deleted_track"])
            if len(playlist["tracks"]) == 0:
                sorted_candidates = [x[0] for x in candidate[1][:1000]]
            else:
                if "number_of_tracks" not in playlist:
                    number_of_tracks = len(playlist["tracks"])
                else:
                    number_of_tracks = int(playlist["number_of_tracks"])
                if "name" not in playlist or len(playlist["name"]) == 0 or playlist["name"] == "#no-name#" or number_of_tracks > 20:
                    sorted_candidates = []
                    for size, model in zip(no_name_sizes, no_name_models):
                        if size == number_of_tracks:
                            sorted_candidates = model.get_top(candidate, playlist)
                            break
                    if len(sorted_candidates) == 0:
                        sorted_candidates = no_name_model.get_top(candidate, playlist)
                else:
                    sorted_candidates = []
                    for size, model in zip(sizes, name_models):
                        if size == number_of_tracks:
                            sorted_candidates = model.get_top(candidate, playlist)
                            break
                    if len(sorted_candidates) == 0:
                        sorted_candidates = name_model.get_top(candidate, playlist)
            result["recommended"] = sorted_candidates
            output_file.write(json.dumps(result) + "\n")
            if i % 20 == 0:
                output_file.flush()


if __name__ == '__main__':
    tfe.enable_eager_execution()
    print(sys.argv)
    playlists_filename = sys.argv[1]
    track_to_album = sys.argv[2]
    album_to_artist = sys.argv[3]
    vector_model_description = sys.argv[4]
    ui_vector_model_description = sys.argv[5]

    track_popularity_filename = sys.argv[6]
    album_popularity_filename = sys.argv[7]
    artist_popularity_filename = sys.argv[8]
    candidate_filename = sys.argv[9]
    catboost_model_file = sys.argv[10]
    output_filename = sys.argv[11]

    svdpp_folder = "/home/alzaharov/name_vectors/stash"
    nals_folder = "/home/alzaharov/name_vectors"


    # skip = int(sys.argv[8])
    # lines_to_do = int(sys.argv[9])

    track_to_album = np.array(utils.load_info_for_model.load_json(track_to_album))
    album_to_artist = np.array(utils.load_info_for_model.load_json(album_to_artist))

    track_popularity = load_popularity(track_popularity_filename)
    album_popularity = load_popularity(album_popularity_filename)
    artist_popularity = load_popularity(artist_popularity_filename)
    with open(playlists_filename) as playlists_file:
        playlists = [json.loads(line) for line in playlists_file]
    candidates = utils.serialize_iterable.load(candidate_filename)
    # print(len(candidates[0][1]))
    # exit(1)
    vector_models, ui_models = utils.load_vector_models.load_models_from_info_file(vector_model_description, VectorModel, ItemVectorModel)
    # ui_models = utils.load_vector_models.load_user_item_models_from_file(ui_vector_model_description, ItemVectorModel)

    m, uim, svd_pp = utils.load_vector_models.load_svd_pp_model(svdpp_folder, VectorModel, ItemVectorModel,
                                                                NumpySvdPP)
    vector_models.append(m)
    ui_models.append(uim)
    m, uim, nals = utils.load_vector_models.load_name_als(nals_folder, VectorModel, ItemVectorModel,
                                                          NumpyNameAls)
    vector_models.append(m)
    ui_models.append(uim)

    name_feature_generator = lambda playlist, candidates: calculate_features.all_features.all_features(playlist, candidates,
                                                                                                  (vector_models,
                                                                                                   ui_models, svd_pp, nals),
                                                                                                  track_to_album,
                                                                                                  album_to_artist,
                                                                                                  [track_popularity,
                                                                                                   album_popularity,
                                                                                                   artist_popularity], 1, playlist["wv"])
    no_name_feature_generator = lambda playlist, candidates: calculate_features.all_features.all_features(playlist,
                                                                                                       candidates,
                                                                                                       (vector_models,
                                                                                                        ui_models, svd_pp, nals),
                                                                                                       track_to_album,
                                                                                                       album_to_artist,
                                                                                                       [
                                                                                                           track_popularity,
                                                                                                           album_popularity,
                                                                                                           artist_popularity],
                                                                                                       0, playlist["wv"])
    sizes = [1,5,10]
    no_name_sizes = [5,10, 25, 100]
    name_models = [CatboostModel(catboost_model_file+"_name_"+str(x) , name_feature_generator) for x in sizes]
    no_name_models = [CatboostModel(catboost_model_file+"_no_name_"+str(x) , no_name_feature_generator) for x in no_name_sizes]
    print("loaded {} models".format(len(name_models)))
    name_model = CatboostModel(catboost_model_file+"_name_10" , name_feature_generator)
    no_name_model = CatboostModel(catboost_model_file+"_no_name_10", no_name_feature_generator)
    print("Model created")
    apply(playlists, candidates, output_filename)
