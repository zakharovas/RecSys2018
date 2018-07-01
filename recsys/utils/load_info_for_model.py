import json
from tqdm import tqdm

def load_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


def load_lines_json(filename):
    output = []
    with open(filename, "r") as file:
        for line in tqdm(file):
            output.append(json.loads(line))
    return output

