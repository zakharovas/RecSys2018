import numpy as np
import re
from tqdm import tqdm
import itertools
import collections
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse

import candidate_generator.candidate_generator


def group(iterable, size):
    args = [iter(iterable)] * size
    return zip(*args)


class RandomWalkCandidates(candidate_generator.candidate_generator.CandidateGenerator):
    def __init__(self, prob, train, candidate_size, visits_in_last, max_playlist_number=1010000):
        self._probabiblity = prob
        self._playlist_to_track = []
        self._track_to_playlist = []
        self._candidate_size = candidate_size
        self._visits_in_last = visits_in_last
        self._track_to_number = dict()
        self._number_to_track = []
        self._number_of_playlists = max_playlist_number
        self._sorted_tracks = []
        self._str = "RW"

        self._build_graph(train)

    def _build_graph(self, train):
        for _ in range(self._number_of_playlists):
            self._playlist_to_track.append([])
        for playlist in tqdm(train):
            for track in playlist["tracks"]:
                if track not in self._track_to_number:
                    self._track_to_number[track] = len(self._number_to_track)
                    self._number_to_track.append(track)
                    self._track_to_playlist.append([])
                self._track_to_playlist[self._track_to_number[track]].append(playlist["pid"])
                self._playlist_to_track[playlist["pid"]].append(self._track_to_number[track])
        self._sorted_tracks = sorted(enumerate(self._track_to_playlist), key=lambda x: len(x[1]), reverse=True)

    def walk(self, playlist):
        start_vertexes = \
            list(map(lambda x: self._track_to_number[x],
                     filter(lambda x: x in self._track_to_number, playlist["tracks"])))
        if len(start_vertexes) == 0:
            return (1, [(self._number_to_track[x[0]], 0) for x in
                        self._sorted_tracks], 0)
        max_step = 20000000
        max_size = 100000
        playlists_random = np.random.randint(0, 1000000, max_size)
        track_random = np.random.randint(0, 10000000, max_size)
        jump = np.random.random_sample(max_size)
        over_border = 0
        start = np.random.randint(0, 1000000, max_size)
        set_of_tracks = set(self._track_to_number[x] for x in playlist["tracks"] if x in self._track_to_number)
        visits = np.zeros(len(self._track_to_playlist))

        vertex = np.random.choice(start_vertexes, size=1)[0]
        i = 0
        while over_border < self._candidate_size:
            playlist = self._track_to_playlist[vertex][playlists_random[i % max_size] % len(self._track_to_playlist[vertex])]
            vertex = self._playlist_to_track[playlist][track_random[i % max_size] % len(self._playlist_to_track[playlist])]
            if vertex not in set_of_tracks:
                visits[vertex] += 1
            if visits[vertex] == self._visits_in_last and vertex not in set_of_tracks:
                over_border += 1
                # print("OVER {}".format(over_border))
            if jump[i % max_size] < self._probabiblity:
                vertex = start_vertexes[start[i % max_size] % len(start_vertexes)]
            i += 1
            if i == max_step:
                print(i)
                break
            if i == max_size:
                playlists_random = np.random.randint(0, 1000000, max_size)
                track_random = np.random.randint(0, 10000000, max_size)
                jump = np.random.random_sample(max_size)
                start = np.random.randint(0, 1000000, max_size)

        return (i, [(self._number_to_track[x[0]], x[1]) for x in
                    sorted(enumerate(visits), key=lambda x: x[1], reverse=True)], 1)

    def candidates_with_features(self, playlist):
        return self.walk(playlist)

    def create_candidates(self, playlist):
        return [x[0] for x in self.walk(playlist)[1]][:self._candidate_size]


class NameRandomWalk(candidate_generator.candidate_generator.CandidateGenerator):

    def __init__(self, prob, train, candidate_size, visits_in_last, max_playlist_number=1010000):
        self._stop_words = set(["is", "are", "be", "am", "to", "the"])
        self._probabiblity = prob
        self._playlist_to_track = []
        self._track_to_playlist = []
        self._candidate_size = candidate_size
        self._visits_in_last = visits_in_last
        self._track_to_number = dict()
        self._word_to_number = dict()
        self._number_to_track = []
        self._number_of_playlists = max_playlist_number
        self._sorted_tracks = []
        self._ngram_size = 3
        self._tf_idf = TfidfTransformer(norm="l1")
        self._build_graph(train)
        self._str = "NRW"

    @staticmethod
    def normalize_name(name):
        name = name.lower()
        name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    @staticmethod
    def extract_words(name, ngram_max_size):
        words = name.split()
        result = [name]
        for ngram_size in range(ngram_max_size):
            for word_group in group(words, ngram_size):
                word = " ".join(word_group)
                if len(word) < 2:# or word in self._stop_words:
                    continue
                result.append(word)
        return result

    def _build_graph(self, train):
        # for _ in range(self._number_of_playlists):
        #     self._playlist_to_track.append([])

        row = []
        col = []
        data = []
        for i,playlist in enumerate(tqdm(train)):
            words = self.extract_words(self.normalize_name(playlist["name"]), self._ngram_size)

            # print(words)
            # print(playlist["name"])
            for word in words:
                if word not in self._word_to_number:
                    self._word_to_number[word] = len(self._word_to_number)
                    self._playlist_to_track.append([])
            word_numbers = [self._word_to_number[x] for x in words]
            counted = collections.Counter(word_numbers)
            for element in counted:
                row.append(i)
                col.append(element)
                data.append(counted[element])
            for track in playlist["tracks"]:
                if track not in self._track_to_number:
                    self._track_to_number[track] = len(self._number_to_track)
                    self._number_to_track.append(track)
                    self._track_to_playlist.append([])
                self._track_to_playlist[self._track_to_number[track]] += word_numbers
                for number in word_numbers:
                    self._playlist_to_track[number].append(self._track_to_number[track])
        self._sorted_tracks = sorted(enumerate(self._track_to_playlist), key=lambda x: len(x[1]), reverse=True)
        matrix = sparse.csr_matrix((data, (row, col)), shape=(len(train), len(self._word_to_number)))
        self._tf_idf.fit(matrix)

    def walk(self, playlist):
        words = self.extract_words(self.normalize_name(playlist["name"]), self._ngram_size)
        start_vertexes = \
            collections.Counter(map(lambda x: self._word_to_number[x],
                     filter(lambda x: x in self._word_to_number, words)))
        row = []
        col = []
        data = []
        for i in start_vertexes:
            row.append(0)
            col.append(i)
            data.append(start_vertexes[i])
        matrix = sparse.csr_matrix((data, (row, col)), shape=(1, len(self._word_to_number)))
        probabilities = self._tf_idf.transform(matrix).todense().flatten()
        vertexes = []
        pr = []
        for i in start_vertexes:
            vertexes.append(i)
            pr.append(probabilities[0,i])

        if len(vertexes) == 0:
            # print("Name Down")
            return (1, [(self._number_to_track[x[0]], 0) for x in
                        self._sorted_tracks], 0)
        # print("Name works")
        max_step = 20000000
        max_size = 100000
        playlists_random = np.random.randint(0, 1000000, max_size)
        track_random = np.random.randint(0, 10000000, max_size)
        jump = np.random.random_sample(max_size)
        over_border = 0
        start = np.random.choice(vertexes, max_size, p=pr)
        set_of_tracks = set(self._track_to_number[x] for x in playlist["tracks"] if x in self._track_to_number)
        visits = np.zeros(len(self._track_to_playlist))

        vertex = start[0]
        i = 1
        while over_border < self._candidate_size:
            vertex = self._playlist_to_track[vertex][track_random[i % max_size] % len(self._playlist_to_track[vertex])]
            if vertex not in set_of_tracks:
                visits[vertex] += 1
            if visits[vertex] == self._visits_in_last and vertex not in set_of_tracks:
                over_border += 1
            vertex = self._track_to_playlist[vertex][playlists_random[i % max_size] % len(self._track_to_playlist[vertex])]
                # print("OVER {}".format(over_border))
            if jump[i % max_size] < self._probabiblity:
                vertex = start[i % max_size]
            i += 1
            if i == max_step:
                print(i)
                break
            if i == max_size:
                playlists_random = np.random.randint(0, 1000000, max_size)
                track_random = np.random.randint(0, 10000000, max_size)
                jump = np.random.random_sample(max_size)
                start = np.random.choice(vertexes, max_size, p=pr)

        return (i, [(self._number_to_track[x[0]], x[1]) for x in
                    sorted(enumerate(visits), key=lambda x: x[1], reverse=True)], 1)

    def candidates_with_features(self, playlist):
        return self.walk(playlist)


    def create_candidates(self, playlist):
        return [x[0] for x in self.walk(playlist)[1]]
