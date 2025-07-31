# 📦 Imports
import streamlit as st
import pandas as pd
import pickle
import json
import re

from surprise import SVD
from surprise import Dataset, Reader
from difflib import get_close_matches

# ⚙️ Streamlit Page Configuration
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# 📌 Load Data
df = pd.read_pickle('movie_df.pkl')
merged = pd.read_csv('merged.csv')
ratings = pd.read_csv('ratings.csv')

# 🧠 Load Models and Matrices
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("cosine_sim_tfidf.pkl", "rb") as f:
    cosine_sim_tfidf = pickle.load(f)
    
with open('cosine_sim_meta.pkl', 'rb') as f:
    cosine_sim_meta = pickle.load(f)

with open('svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

# 🧾 Clean Movie Titles
df['title_clean'] = df['title_clean'].str.lower().str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
indices_tmdb = pd.Series(df.index, index=df['title_clean']).drop_duplicates()

# 📌 Recommendation Functions

def get_content_recommendations(title, df, cosine_sim, top_n=10):
    title_clean = re.sub(r'\s*\(\d{4}\)', '', title).strip().lower()
    
    if 'title_clean' not in df.columns:
        df['title_clean'] = df['title'].str.lower().str.strip()
    
    indices_tmdb = pd.Series(df.index, index=df['title_clean']).to_dict()

    if title_clean not in indices_tmdb:
        return None

    idx = indices_tmdb[title_clean]

    if idx >= cosine_sim.shape[0]:
        return None

    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    result_df = df.iloc[movie_indices][[
        'title_clean', 'overview', 'genres_list', 'director', 'top_cast', 'vote_average', 'id'
    ]].copy()
    result_df['similarity_score'] = [float(s) for s in similarity_scores]

    results = result_df.to_dict('records')
    return results


def get_svd_recommendations(user_id, model, ratings_df, merged_df, top_n=10):
    
    watched = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].values
    
    
    all_movies = merged_df['movie_id'].unique()
    
    
    unseen = [movie for movie in all_movies if movie not in watched]

    
    predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in unseen]
    predictions.sort(key=lambda x: x[1], reverse=True)

    
    top_predictions = predictions[:top_n]
    movie_ids = [movie_id for movie_id, _ in top_predictions]
    scores = [score for _, score in top_predictions]

    
    titles_clean = merged_df[merged_df['movie_id'].isin(movie_ids)] \
        .drop_duplicates('movie_id') \
        .set_index('movie_id') \
        .loc[movie_ids]['title_clean'].values

    return titles_clean, scores
def get_hybrid_recommendations(user_id, movie_ratings, model, final_df,
                               content_weight=0.6, svd_weight=0.4, top_n=10):
    import pandas as pd

    hybrid_scores = {}
    metadata = {}

    # 1. Content-based Recommendations
    movie_ratings = movie_ratings[:5]
    content_scores_dict = {}
    for movie, _ in movie_ratings:
        results = get_content_recommendations(movie, df, cosine_sim_meta)

        if results:
            for r in results:
                title_clean = r['title_clean'].strip().lower()
                try:
                    score = float(r['similarity_score'])
                except:
                    continue
                content_scores_dict[title_clean] = content_scores_dict.get(title_clean, 0) + score

                if title_clean not in metadata:
                    metadata[title_clean] = {
                        'overview': r.get('overview', ''),
                        'genres': r.get('genres_list', ''),
                        'director': r.get('director', ''),
                        'top_cast': r.get('top_cast', ''),
                        'vote_average': r.get('vote_average', None),
                        'id': r.get('id', None)
                    }

    # 2. SVD Recommendations
    svd_titles_clean, svd_scores = get_svd_recommendations(
        user_id,
        model,
        final_df[['user_id', 'movie_id', 'rating']],
        final_df,
        top_n=50
    )

    svd_scores_dict = {}
    for title_clean, score in zip(svd_titles_clean, svd_scores):
        try:
            title_key = title_clean.strip().lower()
            score_value = float(score)
            svd_scores_dict[title_key] = score_value
        except (ValueError, TypeError):
            continue  

    # 3. Adjust weights if no content-based results
    if not content_scores_dict:
        content_weight, svd_weight = 0, 1.0

    total_weight = float(content_weight) + float(svd_weight)

    # 4. Combine all titles
    all_titles = set(content_scores_dict.keys()).union(svd_scores_dict.keys())

    for title_clean in all_titles:
        try:
            content_score = float(content_scores_dict.get(title_clean, 0)) * content_weight
        except:
            content_score = 0.0

        try:
            svd_score = float(svd_scores_dict.get(title_clean, 2.5)) * svd_weight
        except:
            svd_score = 0.0

        hybrid_scores[title_clean] = (content_score + svd_score) / total_weight

    # 5. Sort and collect metadata
    recommendations = pd.Series(hybrid_scores).sort_values(ascending=False).head(top_n)
    result = []
    for title_clean, score in recommendations.items():
        meta = metadata.get(title_clean, {})
        row = {
            'title_clean': title_clean.title(),
            'score': round(score, 2),
            'overview': meta.get('overview', ''),
            'genres': meta.get('genres', ''),
            'director': meta.get('director', ''),
            'top_cast': meta.get('top_cast', ''),
            'vote_average': meta.get('vote_average', None),
            'id': meta.get('id', None)
        }
        result.append(row)

    return result



# 🎬 Streamlit Interface
st.title("🎥 Movie Recommendation System")
st.markdown("Choose the type of recommendation you want:")

option = st.selectbox("🔍 Select Recommendation Type", ['Content-Based', 'SVD', 'Hybrid'])

if option == 'Content-Based':
    movie_title = st.text_input("🎞️ Enter a movie you liked")
    
    if st.button("✨ Get Recommendations"):
        recommendations = get_content_recommendations(movie_title, df, cosine_sim_meta)

        if recommendations:
            st.success(f"Found {len(recommendations)} similar recommendations:")
            for i, movie in enumerate(recommendations):
                st.markdown(f"### {i+1}. 🎬 {movie['title_clean']}")
                st.write(f"**Overview:** {movie['overview']}")
                st.write(f"**Genres:** {movie['genres_list']}")
                st.write(f"**Director:** {movie['director']}")
                st.write(f"**Top Cast:** {movie['top_cast']}")
                st.write(f"**Similarity Score:** {movie['similarity_score']:.4f}")
                st.write(f"**TMDB ID:** {movie['id']}")
                st.markdown("---")
        else:
            st.warning("No recommendations found. Please make sure the movie title is spelled correctly.")

elif option == 'SVD':
    user_id = st.number_input("🧑‍💻 Enter User ID", min_value=1, step=1)
    if st.button("📊 Get Recommendations"):
        titles, scores = get_svd_recommendations(user_id, svd_model, ratings, merged, top_n=10)
        st.success(f"Top recommendations for User {user_id}:")
        for t, s in zip(titles, scores):
            st.write(f"🎬 {t} - ✅ Predicted Rating: {s:.2f}")
            st.markdown("---")

elif option == 'Hybrid':
    user_id = st.number_input("🧑‍💻 Enter User ID", min_value=1, step=1)
    user_movies = st.text_area("🎞️ Movies you liked (separated by commas)", placeholder="e.g. Inception, Titanic, Matrix")

    if st.button("🔀 Get Hybrid Recommendations"):
        movies = [m.strip() for m in user_movies.split(",") if m.strip()]
        
        if len(movies) == 0:
            st.warning("Please enter at least one movie.")
        else:
            fake_ratings = [(m, 5.0) for m in movies]
            hybrid_recommendations = get_hybrid_recommendations(user_id, fake_ratings, svd_model, ratings)

            if not hybrid_recommendations:
                st.warning("❌ No suitable recommendations found.")
            else:
                st.success("📽️ Hybrid Recommendations Based on Your Taste:")
                for rec in hybrid_recommendations:
                    st.write(f"🎬 {rec['title_clean']} — ⭐ {rec['score']:.2f}")

