import streamlit as st
import pickle
import pandas as pd

# Load data (adjust paths if needed)
movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open("similarity.pkl", "rb"))

st.title("🎬 Movie Recommender System")

# Select movie
selected_movie = st.selectbox(
    "Choose a movie:",
    movies['title'].values
)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for i in recommendations:
        st.write("✅", i)
# To run the app, use the command: streamlit run app.py 