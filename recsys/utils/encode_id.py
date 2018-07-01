import sys
import json
from tqdm import tqdm

if __name__ == '__main__':
    playlists_filename = sys.argv[1]
    artists_filename = sys.argv[2]
    albums_filename = sys.argv[3]
    tracks_filename = sys.argv[4]
    al2ar_filename = sys.argv[5]
    tr2al_filename = sys.argv[6]
    artists = dict()
    albums = dict()
    tracks = dict()
    tracks_to_albums = []
    albums_to_artists = []
    with open(playlists_filename) as playlist_file:
        for line in tqdm(playlist_file):
            playlinst = json.loads(line)
            for track in playlinst["tracks"]:
                if track["artist_uri"] not in artists:
                    artists[track["artist_uri"]] = len(artists)
                if track["album_uri"] not in albums:
                    albums[track["album_uri"]] = len(albums)
                    albums_to_artists.append(artists[track["artist_uri"]])
                if track["track_uri"] not in tracks:
                    tracks[track["track_uri"]] = len(tracks)
                    tracks_to_albums.append(albums[track["album_uri"]])

    with open(artists_filename, "w") as artists_file, open(albums_filename, "w") as albums_file, \
            open(tracks_filename, "w") as track_file, open(tr2al_filename, "w") as tr2al_file, \
            open(al2ar_filename, "w") as al2ar_file:
        artists_file.write(json.dumps(artists))
        albums_file.write(json.dumps(albums))
        track_file.write(json.dumps(tracks))
        tr2al_file.write(json.dumps(tracks_to_albums))
        al2ar_file.write(json.dumps(albums_to_artists))

