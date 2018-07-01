from candidate_generator.random_walk import *


class ComplexGenerator:
    def __init__(self, generators, size):
        self._generators = generators
        self._size = size

    def candidates(self, playlist):
        scores = self._create_generators_scores(playlist)
        tracks = self._create_candidates_from_results(scores)
        return self._extract_features(tracks, scores)

    def add_features(self, playlist, tracks):
        scores = self._create_generators_scores(playlist)
        return self._extract_features(tracks, scores)

    def _extend_vectors(self, short_vectors):
        max_size = 2300000
        candidate_scores = [(i, 0) for i in range(max_size)]
        for x in short_vectors[1]:
            candidate_scores[x[0]] = x
        return [short_vectors[0], candidate_scores]

    def _create_generators_scores(self, playlist):
        generation_results = []
        for generator in self._generators:

            generation_results.append((str(generator), self._extend_vectors(generator.candidates_with_features(playlist))))

        return generation_results

    def _create_candidates_from_results(self, scores):
        tracks = set()
        expected_size = self._size
        sorted_scores = [sorted(generator_score[1][1], key=lambda x: x[1], reverse=True) for generator_score in scores]
        iterators = [0] * len(sorted_scores)
        while len(tracks) < expected_size:
            for i, score in enumerate(sorted_scores):
                # if i > 0:
                #     break
                while score[iterators[i]][0] in tracks:
                    iterators[i] += 1
                tracks.add(score[iterators[i]][0])
        return list(tracks)

    def _extract_features(self, candidates, scores):
        sorted_by_id_scores = [[name, score[0], list(sorted(score[1], key=lambda x: x[1], reverse=True))] for name, score in scores]
        for i in range(len(sorted_by_id_scores)):
            sorted_by_id_scores[i][2] = [(x[0],x[1],i) for i, x in enumerate(sorted_by_id_scores[i][2])]
        sorted_by_id_scores = [(score[0], score[1], list(sorted(score[2], key=lambda x: x[0]))) for score in sorted_by_id_scores]
        # name, sum_score, list([track, score])
        result = [[generator[0], generator[1], [generator[2][x] for x in candidates]] for generator in
                  sorted_by_id_scores]
        return result
