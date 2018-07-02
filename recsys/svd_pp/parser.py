import re

from json import loads
from json import dumps

from tqdm import tqdm
from numpy.random import randint

from collections import defaultdict
from itertools import zip_longest


class Encoding:
    """
    Class of encoding: Keys <-> [0, 1, ...]
    """
    def __init__(self):
        self.encode = dict()
        self.decode = list()
        
    def add(self, key):
        if not key in self.encode:
            self.encode[key] = len(self.decode)
            self.decode.append(key)
        return self.encode[key]
    
    def get(self, idx):
        return self.decode[idx]


def grouper(iterable, group_size, fillvalue=None):
    args = [iter(iterable)] * group_size
    return zip_longest(*args, fillvalue=fillvalue)


def parse_name(name, prefix=''):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.split()
    name = [prefix + x for x in name]
    return " ".join(name)

user_encoding = Encoding()
item_encoding = Encoding()

v = defaultdict(set)

print('Counting users and items...')
with open('dataset.json', 'r') as source:
    for line in tqdm(source):
            record = loads(line)

            u = user_encoding.add(parse_name(record['name']))
            for track in record['tracks']:
                v[u].add(item_encoding.add(track))

user_count = len(v)
item_count = len(item_encoding.decode)

group_size = 2048

print('Users: ', user_count)
print('Items: ', item_count)

record_count = 0

print('Encoding users and items...')
with open('encoded_playlists.json', 'w') as target:
    for u in tqdm(v):
        for chunk in grouper(v[u], group_size):
            target.write(dumps({
                'user': u,
                'view': [i for i in chunk if not i is None]
            }))
            target.write('\n')
            record_count += 1
print('Total records done: ', record_count)

print('Saving encodings...')
with open('user_encoding.json', 'w') as target:
    for idx in tqdm(range(user_count)):
        target.write(dumps({
            'key': user_encoding.get(idx),
            'val': idx
        }))
        target.write('\n')

with open('item_encoding.json', 'w') as target:
    for idx in tqdm(range(item_count)):
        target.write(dumps({
            'key': item_encoding.get(idx),
            'val': idx
        }))
        target.write('\n')
