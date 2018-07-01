import json
import numpy as np
from tqdm import tqdm

def dump(iterable, filename):
    print("SAVING TO {}".format(filename))
    with open(filename, "w") as output_file:
        for x in tqdm(iterable):
            output_file.write(json.dumps(x) + "\n")


def load(filename):
    print("LOADING FROM {}".format(filename))
    with open(filename) as input_file:
        return [json.loads(line) for line in tqdm(input_file)]

def dump_matrix(matrix, filename):
    print("SAVING MATRIX TO {}".format(filename))
    with open(filename, "wb") as output_file:
        np.save(output_file, matrix)

def load_matrix(filename):
    print("LOADING MATRIX FROM {}".format(filename))
    return np.load(filename)
