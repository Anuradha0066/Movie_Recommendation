"""
Streamlit Movie Recommender - Text Only Version
by Anuradha
"""

import ast
from difflib import get_close_matches

import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Setup ----------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

# ---------------------- Build Model ----------------------
@st.cache_data
def build_model():
    # Direct paths to CSVs
    movies = pd.read_csv(r'C:\Users\HP\Downloads\tmdb_5000_movies.csv')
    credits = pd.read_csv(r'C:\Users\HP\Downloads\tmdb_5000_credits.csv')

    # Merge and select columns
    movies = movies.merge(credits, on="title")
    movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    # Convert JSON-like text columns to lists
    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    def convert_top3(text):
        L = []
        for idx, i in enumerate(ast.literal_eval(text)):
            if idx < 3:
                L.append(i['name'])
        return L

    def get_director(text):
        for i in ast.literal_eval(text):
            if i.get('job') == 'Director':
                return [i.get('name')]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_top3)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Clean spaces from names
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # Combine all tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x)).str.lower()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join([ps.stem(i) for i in x.split()]))

    # Vectorize and compute similarity
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

# Build the model
with st.spinner("Loading movies..."):
    movies_df, similarity = build_model()
titles = movies_df['title'].values.tolist()

# ---------------------- Fuzzy Match ----------------------
def fuzzy_match(query, choices, n=5, cutoff=0.6):
    if not query:
        return []
    return get_close_matches(query, choices, n=n, cutoff=cutoff)

# ---------------------- Streamlit UI ----------------------
st.title("🎬 Movie Recommender — Text Only Version")
st.write("Search a movie and get top similar movies.")

with st.sidebar:
    st.header("Search")
    movie_input = st.text_input("Type movie title (e.g., Avatar)")
    use_fuzzy = st.checkbox("Enable fuzzy suggestions", value=True)
    n_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    st.markdown("---")

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Your selection")
    if movie_input:
        matches = fuzzy_match(movie_input, titles, n=5, cutoff=0.5) if use_fuzzy else ([movie_input] if movie_input in titles else [])
        if matches:
            choice = st.selectbox("Did you mean:", options=matches + (['Search exact'] if movie_input not in matches else []))
        else:
            choice = st.selectbox("No close matches — try exact title:", options=['Search exact'])
    else:
        choice = st.selectbox("Choose:", options=["Avatar", "The Avengers", "Titanic"])

    st.markdown("---")

    if st.button("Recommend"):
        if choice == 'Search exact' and movie_input:
            selected = movie_input.strip()
        else:
            selected = choice

        if selected not in titles:
            st.error(f"Movie '{selected}' not found.")
        else:
            idx = movies_df[movies_df['title'] == selected].index[0]
            distances = similarity[idx]
            movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n_recs+1]

            st.success(f"Top {n_recs} movies similar to: {selected}")
            for pair in movie_list:
                m_title = movies_df.iloc[pair[0]].title
                st.write(f"- **{m_title}**")

with col2:
    st.subheader("Preview — Selected Movie Tags")
    try:
        preview_title = selected if 'selected' in locals() else titles[0]
        preview_idx = movies_df[movies_df['title'] == preview_title].index[0]
        st.markdown(f"### {preview_title}")
        tags_text = " ".join(movies_df.iloc[preview_idx]['tags'].split()[:80])
        st.write(tags_text + '...')
    except Exception:
        st.info("Choose a movie and press Recommend.")

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ by Anuradha")
