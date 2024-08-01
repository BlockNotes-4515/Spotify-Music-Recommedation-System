#################################################################################################
#################################################################################################
# VERSION 1.0
#################################################################################################
#################################################################################################
# import streamlit as st
# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import cdist
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.cluster import KMeans

# # Load your data
# data = pd.read_csv("data.csv")
# genre_data = pd.read_csv('data_by_genres.csv')
# year_data = pd.read_csv('data_by_year.csv')

# # Define your clustering pipeline
# cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
# X = genre_data.select_dtypes(np.number)
# cluster_pipeline.fit(X)
# genre_data['cluster'] = cluster_pipeline.predict(X)

# # Define the functions used in the recommendation system
# number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
#  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# def flatten_dict_list(dict_list):
#     flattened_dict = defaultdict()
#     for key in dict_list[0].keys():
#         flattened_dict[key] = []
#     for dictionary in dict_list:
#         for key, value in dictionary.items():
#             flattened_dict[key].append(value)
#     return flattened_dict

# def get_song_data(song, spotify_data):
#     try:
#         song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
#         return song_data
#     except IndexError:
#         return None

# def get_mean_vector(song_list, spotify_data):
#     song_vectors = []
#     for song in song_list:
#         song_data = get_song_data(song, spotify_data)
#         if song_data is None:
#             print(f'Warning: {song["name"]} does not exist in the database')
#             continue
#         song_vector = song_data[number_cols].values
#         song_vectors.append(song_vector)
#     song_matrix = np.array(song_vectors)
#     return np.mean(song_matrix, axis=0)

# def recommend_songs(song_list, spotify_data, n_songs=10):
#     metadata_cols = ['name', 'year', 'artists']
#     song_dict = flatten_dict_list(song_list)
#     song_center = get_mean_vector(song_list, spotify_data)
#     scaler = cluster_pipeline.steps[0][1]
#     scaled_data = scaler.transform(spotify_data[number_cols])
#     scaled_song_center = scaler.transform(song_center.reshape(1, -1))
#     distances = cdist(scaled_song_center, scaled_data, 'cosine')
#     index = list(np.argsort(distances)[:, :n_songs][0])
#     rec_songs = spotify_data.iloc[index]
#     rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
#     return rec_songs[metadata_cols].to_dict(orient='records')

# # Streamlit app
# def main():
#     st.title("Spotify Music Recommendation System")

#     st.header("Please register:")
#     name = st.text_input("1. Enter your name:")
#     age = st.number_input("2. Enter your age:", min_value=0)
#     gender = st.selectbox("3. Enter your gender:", ["M", "F"])

#     language_option = st.selectbox("4. Select Language:", ["Hindi", "English", "Other"])
#     song_type_option = st.selectbox("5. Select your preferred song type:", ["Rock", "Romantic", "Neon", "Peace", "Hip-Hop"])

#     if st.button("Submit"):
#         st.success("Registered Successfully!")
#     else:
#         st.warning("Not Registered. Only Free Music Available without Premium")
#         return

#     if 'song_list' not in st.session_state:
#         st.session_state.song_list = []

#     st.header("Enter the songs you like:")
#     with st.form(key="song_form"):
#         song_name = st.text_input("Enter the name of a song you like:", key="song_name")
#         artist_name = st.text_input("Enter the name of the artist:", key="artist_name")
#         song_year = st.number_input("Enter the release year of the song:", min_value=0, key="song_year")
#         add_song = st.form_submit_button("Add Song")

#         if add_song:
#             st.session_state.song_list.append({'name': song_name, 'year': song_year, 'artists': artist_name})
#             st.success("Song added!")

#     if st.session_state.song_list:
#         st.write("Songs you have added:")
#         for song in st.session_state.song_list:
#             st.write(f"{song['name']} by {song['artists']} ({song['year']})")

#         if st.button("Get Recommendations"):
#             recommendations = recommend_songs(st.session_state.song_list, data)
#             st.header("Recommended Songs:")
#             for song in recommendations:
#                 st.write(f"{song['name']} by {song['artists']} ({song['year']})")

# if __name__ == "__main__":
#     main()


#################################################################################################
#################################################################################################
# VERSION 2.0
#################################################################################################
#################################################################################################
# import streamlit as st
# import pandas as pd
# import numpy as np
# from testbongo import recommend_songs, find_song
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
# from collections import defaultdict
# from scipy.spatial.distance import cdist
# import sys
# sys.path.append('C:\\Users\\yugan\\Downloads\\archive')


# # Spotify credentials
# sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="a5ad8b6ab10f4e969227a8b8982d9ecd",
#                                                            client_secret="6f6e351c549346d297c9b76cb14587fd"))

# # Load your dataset
# @st.cache
# def load_data():
#     data = pd.read_csv("data.csv")  # Replace with your dataset path
#     return data

# spotify_data = load_data()

# # Streamlit app
# st.title('Spotify Music Recommendation System')

# st.header("Please register:")
# name = st.text_input("1. Enter your name:")
# age = st.number_input("2. Enter your age:", min_value=0)
# gender = st.selectbox("3. Enter your gender:", ["M", "F"])

# language_option = st.selectbox("4. Select Language:", ["Hindi", "English", "Other"])
# song_type_option = st.selectbox("5. Select your preferred song type:", ["Rock", "Romantic", "Neon", "Peace", "Hip-Hop"])

# if st.button("Submit"):
#     st.success("Registered Successfully!")
# else:
#     st.warning("Not Registered. Only Free Music Available without Premium")
#     st.stop()

# if 'song_list' not in st.session_state:
#     st.session_state.song_list = []

# st.header("Enter the songs you like:")
# with st.form(key="song_form"):
#     song_name = st.text_input("Enter the name of a song you like:", key="song_name")
#     artist_name = st.text_input("Enter the name of the artist:", key="artist_name")
#     song_year = st.number_input("Enter the release year of the song:", min_value=0, key="song_year")
#     add_song = st.form_submit_button("Add Song")

#     if add_song:
#         st.session_state.song_list.append({'name': song_name, 'year': song_year, 'artists': artist_name})
#         st.success("Song added!")

# if st.session_state.song_list:
#     st.write("Songs you have added:")
#     for song in st.session_state.song_list:
#         st.write(f"{song['name']} by {song['artists']} ({song['year']})")

#     if st.button("Get Recommendations"):
#         recommendations = recommend_songs(st.session_state.song_list, spotify_data)
#         st.header("Recommended Songs:")
#         for song in recommendations:
#             st.write(f"{song['name']} by {song['artists']} ({song['year']})")

# if __name__ == "__main__":
#     main()



#################################################################################################
#################################################################################################
# VERSION 3.0
#################################################################################################
#################################################################################################
# import streamlit as st
# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import cdist
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.cluster import KMeans
# import testbongo

# # Load the data
# data = pd.read_csv("data.csv")
# genre_data = pd.read_csv('data_by_genres.csv')
# year_data = pd.read_csv('data_by_year.csv')

# # Define necessary functions
# def flatten_dict_list(dict_list):
#     flattened_dict = defaultdict(list)
#     for dictionary in dict_list:
#         for key, value in dictionary.items():
#             flattened_dict[key].append(value)
#     return flattened_dict

# def get_mean_vector(song_list, spotify_data):
#     song_vectors = []
#     for song in song_list:
#         song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
#         song_vector = song_data[number_cols].values
#         song_vectors.append(song_vector)
#     song_matrix = np.array(song_vectors)
#     return np.mean(song_matrix, axis=0)

# def recommend_songs(song_list, spotify_data, n_songs=10):
#     metadata_cols = ['name', 'year', 'artists']
#     song_dict = flatten_dict_list(song_list)
#     song_center = get_mean_vector(song_list, spotify_data)
#     scaler = song_cluster_pipeline.steps[0][1]
#     scaled_data = scaler.transform(spotify_data[number_cols])
#     scaled_song_center = scaler.transform(song_center.reshape(1, -1))
#     distances = cdist(scaled_song_center, scaled_data, 'cosine')
#     index = list(np.argsort(distances)[:, :n_songs][0])
#     rec_songs = spotify_data.iloc[index]
#     rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
#     return rec_songs[metadata_cols].to_dict(orient='records')

# # Streamlit app
# def main():
#     st.title("Song Recommendation System")
    
#     st.header("Please register:")
#     name = st.text_input("1. Enter your name:")
#     age = st.number_input("2. Enter your age:", min_value=0)
#     gender = st.selectbox("3. Enter your gender:", ["M", "F"])

#     language_option = st.selectbox("4. Select Language:", ["Hindi", "English", "Other"])
#     song_type_option = st.selectbox("5. Select your preferred song type:", ["Rock", "Romantic", "Neon", "Peace", "Hip-Hop"])

#     if st.button("Submit"):
#         st.success("Registered Successfully!")
#     else:
#         st.warning("Not Registered. Only Free Music Available without Premium")
#         return

#     song_list = []
#     st.header("Enter the songs you like:")
#     while True:
#         song_name = st.text_input("Enter the name of a song you like (or type 'done' to finish):")
#         if song_name.lower() == 'done':
#             break
#         artist_name = st.text_input("1. Enter the name of the artist:")
#         song_year = st.number_input("2. Enter the release year of the song:", min_value=0)
#         song_list.append({'name': song_name, 'year': song_year, 'artists': artist_name})

#     if song_list:
#         recommendations = recommend_songs(song_list, data)
#         st.header("Recommended Songs:")
#         for song in recommendations:
#             st.write(f"{song['name']} by {song['artists']} ({song['year']})")

# if __name__ == "__main__":
#     main()




#################################################################################################
#################################################################################################
# VERSION 4.1 ( COMMAND LINE ONLY) 
#################################################################################################
#################################################################################################
# import sys
# import os
# import numpy as np
# import pandas as pd

# import seaborn as sns
# import plotly.express as px
# import matplotlib.pyplot as plt
# # %matplotlib inline

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.metrics import euclidean_distances
# from scipy.spatial.distance import cdist

# import warnings
# warnings.filterwarnings("ignore")

# # Ensure the path to your custom module is correctly set
# module_path = r'C:\Users\yugan\Downloads\archive'
# if module_path not in sys.path:
#     sys.path.append(module_path)

# try:
#     from testbongo import flatten_dict_list, song_cluster_pipeline, get_mean_vector, cdist, number_cols
#     print("Imports successful!")
# except ImportError as e:
#     print(f"ImportError: {e}")



# def recommend_songs( song_list, spotify_data, n_songs=10):

#     metadata_cols = ['name', 'year', 'artists']
#     song_dict = flatten_dict_list(song_list)

#     song_center = get_mean_vector(song_list, spotify_data)
#     scaler = song_cluster_pipeline.steps[0][1]
#     scaled_data = scaler.transform(spotify_data[number_cols])
#     scaled_song_center = scaler.transform(song_center.reshape(1, -1))
#     distances = cdist(scaled_song_center, scaled_data, 'cosine')
#     index = list(np.argsort(distances)[:, :n_songs][0])

#     rec_songs = spotify_data.iloc[index]
#     rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
#     return rec_songs[metadata_cols].to_dict(orient='records')
# # Function to get user input
# def get_user_input():
#     print("\n======================= WELCOME TO THE SPOTIFY (MUSICX) ====================")
#     print("Please register:")
#     name = input("\n 1. Enter your name: ")
#     age = int(input("\n 2. Enter your age: "))
#     gender = input("\n 3. Enter your gender (M/F): ")

#     print("\n 4. Select Language:")
#     print("a. Hindi")
#     print("b. English")
#     print("c. Other");
#     language_option = input("\n 5. Enter your choice (a/b/c): ").strip().lower()
#     language = {
#         'a': 'Hindi',
#         'b': 'English',
#         'c': 'Other'
#     }.get(language_option, 'Other')

#     print("\n 6. Select your preferred song type:")
#     print("a. Rock")
#     print("b. Romantic")
#     print("c. Neon")
#     print("d. Peace")
#     print("e. Hip-Hop")
#     type_option = input("\n 7. Enter your choice (a/b/c/d/e): ").strip().lower()
#     song_type = {
#         'a': 'Rock',
#         'b': 'Romantic',
#         'c': 'Neon',
#         'd': 'Peace',
#         'e': 'Hip-Hop'
#     }.get(type_option, 'Rock')  # Default to 'Rock' if invalid input

#     # Confirmation step
#     print("\n Do you allow to submit (Y/N): ")
#     if input().strip().upper() == 'Y':
#         print("\n============================ Registered Successfully! ======================")
#     else:
#         print("\n Not Registered Only Free Musics Available without Premium")
#         exit()
#     print("\n=================================================================================");
#     print("\n=================================================================================");


#     song_list = []
#     while True:
#         song_name = input("\n ----> Enter the name of a song you like (or type 'done' to finish): ")
#         if song_name.lower() == 'done':
#             break
#         artist_name = input("\n 1. Enter the name of the artist: ")
#         song_year = int(input("\n 2. Enter the release year of the song: "))
#         song_list.append({'name': song_name, 'year': song_year, 'artists': artist_name})
#     return song_list

# data = pd.read_csv('data.csv')
# # Main function
# def main():
#     user_songs = get_user_input()
#     recommendations = recommend_songs(user_songs, data)
#     print("============================= WELCOME TO THE SPOTIFY (MUSICX) ==========================");
#     for song in recommendations:
#         print("\n------------------------------------------------------");
#         print(f"{song['name']} by {song['artists']} ({song['year']})")

# if __name__ == "__main__":
#     main()

#################################################################################################
#################################################################################################
# VERSION 5.10
#################################################################################################
#################################################################################################
# import sys
# import numpy as np
# import pandas as pd
# import streamlit as st

# # Ensure the path to your custom module is correctly set
# module_path = r'C:\Users\yugan\Downloads\archive'
# if module_path not in sys.path:
#     sys.path.append(module_path)

# try:
#     from testbongo import flatten_dict_list, song_cluster_pipeline, get_mean_vector, cdist, number_cols
#     st.success("Imports successful!")
# except ImportError as e:
#     st.error(f"ImportError: {e}")

# def recommend_songs(song_list, spotify_data, n_songs=10):
#     metadata_cols = ['name', 'year', 'artists']
#     song_dict = flatten_dict_list(song_list)

#     song_center = get_mean_vector(song_list, spotify_data)
#     scaler = song_cluster_pipeline.steps[0][1]
#     scaled_data = scaler.transform(spotify_data[number_cols])
#     scaled_song_center = scaler.transform(song_center.reshape(1, -1))
#     distances = cdist(scaled_song_center, scaled_data, 'cosine')
#     index = list(np.argsort(distances)[:, :n_songs][0])

#     rec_songs = spotify_data.iloc[index]
#     rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
#     return rec_songs[metadata_cols].to_dict(orient='records')

# # Function to get user input
# def get_user_input():
#     st.title("WELCOME TO THE SPOTIFY (MUSICX)")
#     st.subheader("Please register:")
#     name = st.text_input("1. Enter your name:", key="name")
#     age = st.number_input("2. Enter your age:", min_value=0, key="age")
#     gender = st.selectbox("3. Enter your gender:", ["M", "F"], key="gender")

#     language_option = st.selectbox("4. Select Language:", ["Hindi", "English", "Other"], key="language_option")
#     language = {
#         'Hindi': 'Hindi',
#         'English': 'English',
#         'Other': 'Other'
#     }.get(language_option, 'Other')

#     song_type_option = st.selectbox("6. Select your preferred song type:", ["Rock", "Romantic", "Neon", "Peace", "Hip-Hop"], key="song_type_option")
#     song_type = {
#         'Rock': 'Rock',
#         'Romantic': 'Romantic',
#         'Neon': 'Neon',
#         'Peace': 'Peace',
#         'Hip-Hop': 'Hip-Hop'
#     }.get(song_type_option, 'Rock')  # Default to 'Rock' if invalid input

#     if st.button("Submit"):
#         st.success("Registered Successfully!")
#         return True
#     else:
#         st.warning("Not Registered. Only Free Musics Available without Premium")
#         st.stop()

# data = pd.read_csv('data.csv')

# # Main function
# def main():
#     if 'song_list' not in st.session_state:
#         st.session_state.song_list = []

#     if get_user_input():
#         st.subheader("Add your favorite songs:")
#         add_more = True
#         while add_more:
#             with st.form(key='song_form'):
#                 song_name = st.text_input("Enter the name of a song you like:", key="song_name")
#                 artist_name = st.text_input("Enter the name of the artist:", key="artist_name")
#                 song_year = st.number_input("Enter the release year of the song:", min_value=0, key="song_year")
#                 submit_button = st.form_submit_button(label='Add Song')

#             if submit_button:
#                 st.session_state.song_list.append({'name': song_name, 'year': song_year, 'artists': artist_name})
#                 st.success(f"Added {song_name} by {artist_name} ({song_year})")

#             # Display the current list of songs
#             if st.session_state.song_list:
#                 st.write("Current list of songs:")
#                 st.table(st.session_state.song_list)

#             add_more = st.button("Add another song")

#         if st.button("Get Recommendations"):
#             if st.session_state.song_list:
#                 st.write("Generating recommendations...")  # Debug statement
#                 recommendations = recommend_songs(st.session_state.song_list, data)
#                 st.write("Recommendations generated.")  # Debug statement
#                 st.title("Recommended Songs")
#                 st.table(recommendations)
#             else:
#                 st.warning("Please add at least one song to get recommendations.")

# if __name__ == "__main__":
#     main()


#################################################################################################
#################################################################################################
# VERSION 6 == 5.10 
#################################################################################################
#################################################################################################

import sys
import numpy as np
import pandas as pd
import streamlit as st

# Ensure the path to your custom module is correctly set
module_path = r'C:\Users\yugan\Downloads\archive'
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from testbongo import flatten_dict_list, song_cluster_pipeline, get_mean_vector, cdist, number_cols
    pass 
except ImportError as e:
    st.error(f"ImportError: {e}")

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

# Function to get user input
def get_user_input():
    st.title("WELCOME TO THE SPOTIFY (MUSICX)")
    st.subheader("Please register:")
    name = st.text_input("1. Enter your name:", key="name")
    age = st.number_input("2. Enter your age:", min_value=0, key="age")
    gender = st.selectbox("3. Enter your gender:", ["M", "F"], key="gender")

    language_option = st.selectbox("4. Select Language:", ["Hindi", "English", "Other"], key="language_option")
    language = {
        'Hindi': 'Hindi',
        'English': 'English',
        'Other': 'Other'
    }.get(language_option, 'Other')

    song_type_option = st.selectbox("6. Select your preferred song type:", ["Rock", "Romantic", "Neon", "Peace", "Hip-Hop"], key="song_type_option")
    song_type = {
        'Rock': 'Rock',
        'Romantic': 'Romantic',
        'Neon': 'Neon',
        'Peace': 'Peace',
        'Hip-Hop': 'Hip-Hop'
    }.get(song_type_option, 'Rock')  # Default to 'Rock' if invalid input

    if st.button("Submit"):
        st.success("Registered Successfully!")
        st.session_state.registered = True
    else:
        st.warning("Not Registered. Only Free Musics Available without Premium")
        st.stop()

data = pd.read_csv('data.csv')

# Main function
def main():
    if 'registered' not in st.session_state:
        st.session_state.registered = False

    if 'song_list' not in st.session_state:
        st.session_state.song_list = []

    if not st.session_state.registered:
        get_user_input()
    else:
        st.subheader("Add your favorite songs:")
        add_more = True
        while add_more:
            with st.form(key='song_form'):
                song_name = st.text_input("Enter the name of a song you like:", key="song_name")
                artist_name = st.text_input("Enter the name of the artist:", key="artist_name")
                song_year = st.number_input("Enter the release year of the song:", min_value=0, key="song_year")
                submit_button = st.form_submit_button(label='Add Song')

            if submit_button:
                st.session_state.song_list.append({'name': song_name, 'year': song_year, 'artists': artist_name})
                st.success(f"Added {song_name} by {artist_name} ({song_year})")

            # Display the current list of songs
            if st.session_state.song_list:
                st.write("Current list of songs:")
                st.table(st.session_state.song_list)

            add_more = st.button("Add another song")

        if st.button("Get Recommendations"):
            if st.session_state.song_list:
                st.write("Generating recommendations...")  # Debug statement
                recommendations = recommend_songs(st.session_state.song_list, data)
                st.write("Recommendations generated.")  # Debug statement
                st.title("Recommended Songs")
                st.table(recommendations)
            else:
                st.warning("Please add at least one song to get recommendations.")

if __name__ == "__main__":
    main()
