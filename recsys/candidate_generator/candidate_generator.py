from abc import ABC, abstractmethod


class CandidateGenerator(ABC):

    @abstractmethod
    def create_candidates(self, playlist):
        pass

    def __str__(self):
        return self._str


class AllGenerator(CandidateGenerator):

    def __init__(self, tracks_to_artist):
        self._tracks = tracks_to_artist.keys()

    def create_candidates(self, playlist):
        return self._tracks
