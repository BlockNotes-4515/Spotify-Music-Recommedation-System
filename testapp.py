import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import pickle
import testbongo
import spotipy as sp
from collections import defaultdict
import os;
# Spotify credentials
os.environ["SPOTIFY_CLIENT_ID"] = "a5ad8b6ab10f4e969227a8b8982d9ecd";
os.environ["SPOTIFY_CLIENT_SECRET"] = "6f6e351c549346d297c9b76cb14587fd";
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))
# Load your pre-trained scaler and pipeline
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('song_cluster_pipeline.pkl', 'rb') as f:
    song_cluster_pipeline = pickle.load(f)


def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


# Function to flatten dictionary list
def flatten_dict_list(dict_list):
    flattened_dict = {}
    for key in dict_list[0].keys():
        flattened_dict[key] = [d[key] for d in dict_list]
    return flattened_dict

def get_song_data(song, spotify_data):

    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['year'])

# Function to get mean vector
def get_mean_vector(song_list, spotify_data):

    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

# Recommendation function
def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

# Streamlit interface
def main():
    st.title("Spotify Music Recommendation System")
    st.header("Welcome to the Spotify (MusicX)")

    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=0)
    gender = st.selectbox("Select your gender:", ["M", "F"])

    language = st.selectbox("Select Language:", ["Hindi", "English", "Other"])
    song_type = st.selectbox("Select your preferred song type:", ["Rock", "Romantic", "Neon", "Peace", "Hip-Hop"])

    if st.button("Submit"):
        st.success("Registered Successfully!")

        song_list = []
        song_count = 0
        while True:
            song_name = st.text_input(f"Enter the name of a song you like (or type 'done' to finish):", key=f"song_name_{song_count}")
            if song_name.lower() == 'done':
                break
            artist_name = st.text_input("Enter the name of the artist:", key=f"artist_name_{song_count}")
            song_year = st.number_input("Enter the release year of the song:", min_value=0, key=f"song_year_{song_count}")
            song_list.append({'name': song_name, 'year': song_year, 'artists': artist_name})
            song_count += 1

            if not st.checkbox("Do you want to add another song?", key=f"continue_{song_count}"):
                break

        if song_list:
            data = pd.read_csv('data.csv')  # Load your Spotify data
            recommendations = recommend_songs(song_list, data)
            st.header("Recommended Songs:")
            for song in recommendations:
                st.write(f"{song['name']} by {song['artists']} ({song['year']})")

if __name__ == "__main__":
    main()
