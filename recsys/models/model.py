from abc import ABCMeta, abstractmethod
import random


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._name = ""

    @abstractmethod
    def get_top(self, candidates, playlist):
        pass

    def get_position(self, candidates, playlist):
        top = self.get_top(candidates, playlist)
        if playlist["deleted_track"] in set(top):
            return [i for i, x in enumerate(top) if x == playlist["deleted_track"]][0]
        else:
            return -len(top)

    def __str__(self):
        return self._name


class RandomModel(Model):
    def __init__(self, *args, **kwargs):
        super(RandomModel, self).__init__()
        self._name = "random model"

    def get_top(self, candidates, playlist):
        random.shuffle(candidates)
        return candidates

    def get_position(self, candidates, playlist):
        if playlist["deleted_track"] in set(candidates):
            return random.randint(0, len(candidates))
        else:
            return -len(candidates)

    def get_scores(self, candidates, playlist):
        return [0] * (len(candidates))

    def track_with_track(self, track0, track1):
        return 0
