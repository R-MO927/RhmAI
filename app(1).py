import streamlit as st
import pandas as pd
import pickle
import re
from collections import Counter
from surprise import SVD
from difflib import get_close_matches

# ⚙️ Streamlit Page Configuration
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ---- Dark custom styling ----
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f111a;
        color: #e3e8ff;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f0f9ff;
    }
    .stButton>button {
        background: linear-gradient(135deg,#5c7cfa,#845ef7);
        color: white;
        border: none;
        box-shadow: 0 4px 14px rgba(132,94,247,0.4);
    }
    
    .stSelectbox>div>div>div>div {
        background-color: #1f2340;
        color: #e3e8ff;
        border: 1px solid #5c7cfa;
        border-radius: 8px;
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #1e2445; 
        color: #e3e8ff;
        border: 1px solid #5c7cfa;
        border-radius: 8px;
        padding: 8px;
        font-size: 14px;
    }
    
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        outline: none;
        box-shadow: 0 0 8px rgba(132,94,247,0.6);
        border: 1px solid #a78bfa;
    }
    
    /* placeholder  */
    .stTextInput>div>div>input::placeholder,
    .stNumberInput>div>div>input::placeholder,
    .stTextArea>div>div>textarea::placeholder {
        color: #b0b9ff;
    }
    
    .stMetric>div {
        background: #1f255f;
        border-radius: 12px;
        padding: 10px;
    }
    .stMarkdown pre {
        background: #1f2340;
        color: #c7d2fe;
    }
    </style>

    """,
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown("<h1 style='color:#c7d2fe;'>🎥 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#d0d9ff;'>Choose the type of recommendation you want:</p>", unsafe_allow_html=True)

# ---- Load Resources (cached) ----
@st.cache_data
def load_resources():
    df = pd.read_pickle('movie_df.pkl')
    merged = pd.read_csv('merged.csv')
    ratings = pd.read_csv('ratings.csv')

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

    return df, merged, ratings, tfidf, tfidf_matrix, cosine_sim_tfidf, cosine_sim_meta, svd_model

df, merged, ratings, tfidf, tfidf_matrix, cosine_sim_tfidf, cosine_sim_meta, svd_model = load_resources()

# ---- Preprocess titles once ----
df['title_clean'] = df['title_clean'].str.lower().str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
merged['title_clean'] = merged.get('title_clean', merged.get('title', '')).astype(str).str.lower().str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

# Build lookup series/dicts
indices_tmdb = pd.Series(df.index, index=df['title_clean']).drop_duplicates()
# For hybrid/SVD we use merged with movie_id --> title_clean mapping
movieid_to_titleclean = merged.drop_duplicates('movie_id').set_index('movie_id')['title_clean'].to_dict()

# ---- Recommendation Functions ----

def get_content_recommendations(title, df, cosine_sim, top_n=10):
    title_clean = re.sub(r'\s*\(\d{4}\)', '', title).strip().lower()
    # ensure title_clean column exists
    if 'title_clean' not in df.columns:
        df['title_clean'] = df['title'].str.lower().str.strip()
    indices = pd.Series(df.index, index=df['title_clean']).to_dict()

    if title_clean not in indices:
        return None

    idx = indices[title_clean]
    if idx >= cosine_sim.shape[0]:
        return None

    sim_scores = sorted(
        list(enumerate(cosine_sim[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]

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

    # retrieve cleaned titles in same order
    titles_clean = merged_df[merged_df['movie_id'].isin(movie_ids)] \
        .drop_duplicates('movie_id') \
        .set_index('movie_id') \
        .loc[movie_ids]['title_clean'].values

    return titles_clean, scores

def get_hybrid_recommendations(user_id, movie_ratings, model, final_df,
                               content_weight=0.6, svd_weight=0.4, top_n=10):
    hybrid_scores = {}
    metadata = {}

    # 1. Content-based part (limit to first 5 user-provided)
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

    # 2. SVD part
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

    # 3. Fallback weights
    if not content_scores_dict:
        content_weight, svd_weight = 0, 1.0
    total_weight = float(content_weight) + float(svd_weight)

    # 4. Combine
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

    # 5. Assemble results
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

# ---- Interface ----
option = st.selectbox("🔍 Select Recommendation Type", ['Content-Based', 'SVD', 'Hybrid'])
accent_color = "#845ef7"  
if option == 'Content-Based':
    movie_title = st.text_input("🎞️ Enter a movie you liked", placeholder="e.g. The Matrix")
    if st.button("✨ Get Recommendations"):
        recommendations = get_content_recommendations(movie_title, df, cosine_sim_meta)
        if recommendations:
            st.success(f"Found {len(recommendations)} similar recommendations:")
            for i, movie in enumerate(recommendations):
                st.markdown(
                    f"### {i+1}. 🎬 <span style='color:{accent_color};'>{movie['title_clean'].title()}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"<b style='color:#ffffff;'>Overview:</b> {movie['overview']}", unsafe_allow_html=True)
                st.markdown(f"<b style='color:#ffffff;'>Genres:</b> {movie.get('genres_list', '')}", unsafe_allow_html=True)
                st.markdown(f"<b style='color:#ffffff;'>Director:</b> {movie.get('director', '')}", unsafe_allow_html=True)
                st.markdown(f"<b style='color:#ffffff;'>Top Cast:</b> {movie.get('top_cast', '')}", unsafe_allow_html=True)
                st.markdown(f"<b style='color:#ffdd57;'>Similarity Score:</b> {movie['similarity_score']:.4f}", unsafe_allow_html=True)
                st.markdown(f"<b style='color:#a5d8ff;'>TMDB ID:</b> {movie['id']}", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning("No recommendations found. Please make sure the movie title is spelled correctly.")

elif option == 'SVD':
    user_id = st.number_input("🧑‍💻 Enter User ID", min_value=1, step=1)
    if st.button("📊 Get Recommendations"):
        titles, scores = get_svd_recommendations(user_id, svd_model, ratings, merged, top_n=10)
        st.success(f"Top recommendations for User {user_id}:")
        for t, s in zip(titles, scores):
            st.markdown(
                f"🎬 <span style='color:{accent_color};'>{t}</span> — ✅ <span style='color:#ffdd57;'>Predicted Rating:</span> {s:.2f}",
                unsafe_allow_html=True
            )
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
                    st.markdown(
                        f"🎬 <span style='color:{accent_color};'>{rec['title_clean']}</span> — ⭐ <span style='color:#ffdd57;'>{rec['score']:.2f}</span>",
                        unsafe_allow_html=True
                    )
