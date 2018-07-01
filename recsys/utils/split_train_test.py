import sys
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    input_filename = sys.argv[1]
    train_filename = sys.argv[2]
    test_filename = sys.argv[3]
    train_part = float(sys.argv[4])
    ids = dict()

    with open(input_filename) as input_file, \
         open(train_filename, "w") as train_file, \
         open(test_filename, "w") as test_file:

        for i, line in enumerate(tqdm(input_file)):
            if i == 0:
                train_file.write(line)
                test_file.write(line)
                continue
            id = int(line.split()[1])
            if id not in ids:
                if np.random.random(1)[0] <= train_part:
                    ids[id] = 0
                else:
                    ids[id] = 1
            if ids[id] == 0:
                train_file.write(line)
            else:
                test_file.write(line)
