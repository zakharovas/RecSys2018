from tqdm import tqdm

from models.vector_model import VectorModel
from utils import  serialize_iterable
from utils import  load_info_for_model
import numpy as np

def load_models(filenames, names, class_, ui_class):
    print("READING VECTOR MODELS")
    models = []
    ui_models = []
    for filename, name in tqdm(list(zip(filenames, names))):
        matrix = serialize_iterable.load_matrix(filename)
        model = class_(name)
        ui_model = ui_class("ui" + name)
        model.load_from_dict(matrix)
        ui_model.load_from_dict(matrix)
        models.append(model)
        ui_models.append(ui_model)
    return models, ui_models


def load_models_from_info_file(filename, class_, ui_class):
    names = []
    filenames = []

    with open(filename) as info_file:
        for line in info_file:
            name, filename = line.rstrip().split()
            names.append(name)
            filenames.append(filename)
    return load_models(filenames, names, class_, ui_class)


def load_user_item_models(filenames, names, class_):
    models = []
    for (playlist_filename, item_filename), name in tqdm(list(zip(filenames, names))):
        model = class_(name)
        model.load_from_file(playlist_filename, item_filename)
        models.append(model)
    return models


def load_user_item_models_from_file(filename, class_):
    names = []
    filenames = []
    with open(filename) as info_file:
        for line in info_file:
            name, playlist_filename, track_filename = line.rstrip().split()
            names.append(name)
            filenames.append((playlist_filename, track_filename))
    return load_user_item_models(filenames, names, class_)

def load_svd_pp_model(folder, class_, ui_class, svdclass):
    item_json = load_info_for_model.load_lines_json("{}/item_factor.json".format(folder))
    name_predict =load_info_for_model.load_lines_json("{}/user_factor.json".format(folder))
    item_matrix = np.zeros((2500000, 32))
    inner_matrix = np.zeros((2500000, 32))
    for element in tqdm(item_json):
        item_matrix[element["id"]] = np.array(element["factor"])
        inner_matrix[element["id"]] = np.array(element["inner"])
    item_bias = np.zeros(2500000)
    for element in tqdm(item_json):
        item_bias[element["id"]] = element["bias"]

    name_dict = dict()
    for element in tqdm(name_predict):
        name_dict[element["name"]] = np.array(element["factor"])
    model = class_("svd_vec")
    model.load_from_dict(item_matrix)
    ui_model = ui_class("svd_ui_vec")
    ui_model.load_from_dict(item_matrix)
    svd_model = svdclass("svdpp")
    svd_model.load_from_dict(name_dict, item_matrix, inner_matrix, item_bias)
    return model, ui_model, svd_model



def load_name_als(folder, class_, ui_class, nalsclass):
    item_matrix = serialize_iterable.load_matrix("{}/itest.npy".format(folder))
    name_matrix = serialize_iterable.load_matrix("{}/utest.npy".format(folder))
    name_encoding = load_info_for_model.load_json("{}/encoding.json".format(folder))
    model = class_("name_als_vec")
    model.load_from_dict(item_matrix)
    ui_model = ui_class("name_als_ui_vec")
    ui_model.load_from_dict(item_matrix)
    nals_model = nalsclass("name_als")
    nals_model.load_from_dict(name_matrix, item_matrix, name_encoding)
    return model, ui_model, nals_model


