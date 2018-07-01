import json
import sys
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    with open(sys.argv[1]) as test_file, open(sys.argv[2], 'w') as output_file:
        test = [json.loads(x) for x in test_file]
        sizes_to_delete = np.random.randint(0, 1000, len(test))
        for i, playlist in enumerate(tqdm(test)):
            if len(playlist["tracks"]) <= 25:
                size_to_delete = max(1, (len(playlist["tracks"]) - 10))
            else:
                size_to_delete = len(playlist["tracks"]) - 25
            positions = np.random.choice(len(playlist["tracks"]), size_to_delete, replace=False)
            playlist["deleted_track"] = [playlist["tracks"][position] for position in positions]
            for position in sorted(positions, reverse=True):
                del playlist["tracks"][position]
            output_file.write(json.dumps(playlist) + "\n")
