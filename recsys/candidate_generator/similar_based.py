import candidate_generator.candidate_generator


class SimilarBasedCandidateGenerator(candidate_generator.candidate_generator.CandidateGenerator):

    def __init__(self, similar_artists, tracks_to_artist, artist_to_tracks):
        self._similar_artists = similar_artists
        self._tracks_to_artist = tracks_to_artist
        self._artist_to_tracks = artist_to_tracks

    def create_candidates(self, playlist):
        possible_artists = set()
        tracks_in_playlist = set(playlist["tracks"])
        for track in playlist["tracks"]:
            if track not in self._tracks_to_artist:
                continue
            track_artists = self._tracks_to_artist[track]
            possible_track_artists = track_artists[:]
            for artist in track_artists:
                if artist not in self._similar_artists:
                    continue
                possible_track_artists += self._similar_artists[artist]
            for artist in possible_track_artists:
                possible_artists.add(artist)
        candidates = []
        for artist in possible_artists:
            if artist not in self._artist_to_tracks:
                continue
            candidates += self._artist_to_tracks[artist]
        candidates = list(filter(lambda x: x not in tracks_in_playlist, candidates))
        if len(candidates) == 0:
            candidates = ["empty"]
        return list(set(candidates))
