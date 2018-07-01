import utils.serialize_iterable

import tensorflow.contrib.eager as tfe
import sys
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    tfe.enable_eager_execution()
    with open(sys.argv[1]) as cand_file, open(sys.argv[2], "r") as model_file, open(sys.argv[3], "w") as pl_file:
        names = dict()
        model = dict()
        size = -1
        for i,line in enumerate(tqdm(model_file)):
            if line[0] != "_":
                continue
            model[line.split()[0]] = np.array([float(x) for x in line.rstrip().split()[1:]])
            if size == -1:
                size = len(model[line.split()[0]])
            # if i == 100000:
            #     break
        names_vectors = []
        decrypt = []
        for line in tqdm(cand_file):
            if line in names:
                continue
            words = 0.0
            vector = np.zeros(size)

            for word in line.rstrip().split():
                if word in model:
                    words += 1.0
                    vector += model[word]
            if words > 0:
                vector /= words ** 0.5
            names[line] = len(names_vectors)
            decrypt.append(line)
            names_vectors.append(vector)
        names_vectors = np.array(names_vectors)
        # names_vectors = names_vectors.tolist()
        utils.serialize_iterable.dump_matrix(names_vectors, sys.argv[3])
        # with open(sys.argv[3], "wb") as output_file:
        #     pickle.dump(names_vectors, output_file)
