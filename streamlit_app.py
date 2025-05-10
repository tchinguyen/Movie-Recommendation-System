import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime

# ---------- Load Artifacts ----------
with open("models/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)
with open("models/movie_encoder.pkl", "rb") as f:
    movie_encoder = pickle.load(f)
with open("models/movie_idx_to_genre.pkl", "rb") as f:
    movie_idx_to_genre = pickle.load(f)

model = tf.keras.models.load_model("models/hybrid_ncf_model.h5")

# ---------- Load Movie Metadata ----------
movies_df = pd.read_csv("data/ml-32m/movies.csv")
known_movie_ids = set(movie_encoder.classes_)
movies_df = movies_df[movies_df["movieId"].isin(known_movie_ids)]
movies_df["movie_idx"] = movie_encoder.transform(movies_df["movieId"])
movies_df = movies_df.sort_values("title").reset_index(drop=True)

# Extract year from title for filtering
movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)$")
movies_df["year"] = pd.to_numeric(movies_df["year"], errors="coerce")

# Genre list
all_genres = sorted(set(g for genres in movies_df["genres"].str.split("|") for g in genres if g != "(no genres listed)"))

# ---------- Load Rating Data Safely ----------
ratings_df = pd.read_csv("data/ml-32m/ratings.csv")

# Filter users and movies to match what encoder saw
known_user_ids = set(user_encoder.classes_)
known_movie_ids = set(movie_encoder.classes_)

ratings_df = ratings_df[
    ratings_df["userId"].isin(known_user_ids) &
    ratings_df["movieId"].isin(known_movie_ids)
]

ratings_df["user_idx"] = user_encoder.transform(ratings_df["userId"])
ratings_df["movie_idx"] = movie_encoder.transform(ratings_df["movieId"])

# ---------- Page Setup ----------
st.set_page_config(
    page_title="MovieLens Recommender", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E3B55;
        --secondary-color: #7B8ab6;
        --accent-color: #FFA500;
        --text-color: #333333;
        --light-bg: #f8f9fa;
        --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Body and text styles */
    body {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        color: var(--text-color);
    }
    
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 700;
    }
    
    /* Header area */
    .header-container {
        display: flex;
        align-items: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid var(--primary-color);
        margin-bottom: 2rem;
    }
    
    .logo-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--primary-color);
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: var(--secondary-color);
        font-size: 1.1rem;
        margin-top: 0.3rem;
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: var(--card-shadow);
        margin-bottom: 1.2rem;
        border-left: 4px solid var(--primary-color);
        transition: all 0.25s ease;
    }
    
    .recommendation-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        transform: translateY(-3px);
    }
    
    .movie-title {
        color: var(--primary-color);
        font-size: 1.3rem;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    
    .movie-genres {
        color: var(--secondary-color);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .score-container {
        margin-top: 0.8rem;
    }
    
    .score-bar {
        height: 8px;
        width: 100%;
        background-color: #e9ecef;
        border-radius: 4px;
        margin-top: 5px;
        overflow: hidden;
    }
    
    .score-fill {
        height: 100%;
        background: linear-gradient(to right, var(--secondary-color), var(--primary-color));
        border-radius: 4px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--light-bg);
    }
    
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
        color: var(--primary-color);
    }
    
    /* Info box */
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-color);
        margin-bottom: 1.5rem;
    }
    
    /* Loading animation */
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    /* Button styling */
    .recommend-button {
        background-color: var(--primary-color);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 4px;
        border: none;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .recommend-button:hover {
        background-color: #394b70;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Header Section ----------
st.markdown("""
    <div class="header-container">
        <div>
            <h1 class="logo-text">🎬 CineMatch</h1>
            <p class="subtitle">Intelligent Movie Recommendation Engine</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------- App Description ----------
with st.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
        **CineMatch** uses a sophisticated neural collaborative filtering model to predict which movies 
        will match your taste. The system has been trained on millions of ratings from the MovieLens dataset.
        
        **How it works:**
        1. Select a user profile from the sidebar
        2. Apply optional filters like genre and year range
        3. Click "Generate Recommendations" to see your personalized movie suggestions
        4. Each recommendation includes a match score showing how well it fits your taste
        
        The recommendations are based on both collaborative filtering (what similar users enjoyed) and 
        content-based features (movie genres and characteristics).
    """)

# ---------- Sidebar Filters ----------
st.sidebar.markdown('<div class="sidebar-title">🎯 Recommendation Settings</div>', unsafe_allow_html=True)

# User selection with more context
total_users = len(user_encoder.classes_)
user_idx = st.sidebar.number_input(
    "👤 Select User ID", 
    min_value=0, 
    max_value=total_users-1, 
    value=min(42, total_users-1),
    help=f"Choose from {total_users} available user profiles"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filter Options")

# Genre filter with categorized selection
genre_categories = {
    "Popular": ["Action", "Comedy", "Drama", "Adventure", "Romance"],
    "Specialized": ["Animation", "Fantasy", "Sci-Fi", "Horror", "Thriller"],
    "Niche": ["Documentary", "Western", "Musical", "War", "Film-Noir", "IMAX"]
}

genre_filter_type = st.sidebar.radio("Genre Selection", ["All Genres", "Select Genres", "Category Based"])

selected_genres = []
if genre_filter_type == "Select Genres":
    selected_genres = st.sidebar.multiselect("Choose Specific Genres", all_genres)
elif genre_filter_type == "Category Based":
    genre_category = st.sidebar.selectbox("Choose Genre Category", list(genre_categories.keys()))
    selected_genres = genre_categories[genre_category]
    st.sidebar.markdown(f"*Selected: {', '.join(selected_genres)}*")

# Year range filter
year_min = int(movies_df["year"].min()) if not pd.isna(movies_df["year"].min()) else 1900
year_max = int(movies_df["year"].max()) if not pd.isna(movies_df["year"].max()) else datetime.now().year

year_range = st.sidebar.slider(
    "📅 Release Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# Exclude already rated option
exclude_rated = st.sidebar.checkbox("🚫 Exclude movies already rated", value=True)

# Recommendation count
rec_count = st.sidebar.slider("📊 Number of recommendations", min_value=5, max_value=20, value=10)

st.sidebar.markdown("---")

# User stats
user_ratings = ratings_df[ratings_df["user_idx"] == user_idx]
if not user_ratings.empty:
    avg_rating = user_ratings["rating"].mean()
    num_ratings = len(user_ratings)
    
    st.sidebar.markdown("### 📊 User Profile Stats")
    st.sidebar.markdown(f"**Total ratings:** {num_ratings}")
    st.sidebar.markdown(f"**Average rating:** {avg_rating:.2f}/5.0")
    
    # Find top genres for this user
    user_movie_indices = set(user_ratings["movie_idx"])
    user_movies = movies_df[movies_df["movie_idx"].isin(user_movie_indices)]
    
    genre_counts = {}
    for genres in user_movies["genres"]:
        for genre in genres.split("|"):
            if genre != "(no genres listed)":
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    if genre_counts:
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        st.sidebar.markdown("**Top genres watched:**")
        for genre, count in top_genres:
            st.sidebar.markdown(f"- {genre}: {count} movies")

# ---------- Filter Movies ----------
filtered_movies = movies_df.copy()

# Apply genre filter
if selected_genres:
    genre_filter = filtered_movies["genres"].apply(lambda g: any(genre in g.split("|") for genre in selected_genres))
    filtered_movies = filtered_movies[genre_filter]

# Apply year filter
filtered_movies = filtered_movies[
    (filtered_movies["year"] >= year_range[0]) & 
    (filtered_movies["year"] <= year_range[1])
]

# Exclude rated movies if selected
if exclude_rated:
    rated_movie_idxs = set(ratings_df[ratings_df["user_idx"] == user_idx]["movie_idx"])
    filtered_movies = filtered_movies[~filtered_movies["movie_idx"].isin(rated_movie_idxs)]

# ---------- Predict and Display ----------
if st.sidebar.button("✨ Generate Recommendations", key="recommend_button"):
    if len(filtered_movies) == 0:
        st.warning("⚠️ No movies match your current filters. Please try adjusting your criteria.")
    else:
        with st.spinner("🔍 Analyzing your taste profile and generating personalized recommendations..."):
            try:
                # Display how many movies are being considered
                total_candidates = len(filtered_movies)
                st.info(f"Analyzing {total_candidates} movies that match your filters...")
                
                # Create progress bar for the prediction process
                progress_bar = st.progress(0)
                
                # Prepare inputs
                user_inputs = np.full(len(filtered_movies), user_idx)
                movie_inputs = filtered_movies["movie_idx"].values
                genre_inputs = np.array([
                    movie_idx_to_genre.get(idx, np.zeros(18)) for idx in movie_inputs
                ])
                
                # Update progress
                progress_bar.progress(30)
                
                # Make predictions
                predictions = model.predict({
                    "user_input": user_inputs,
                    "item_input": movie_inputs,
                    "genre_input": genre_inputs
                }, verbose=0).flatten()
                
                # Update progress
                progress_bar.progress(70)
                
                # Sort and get top recommendations
                filtered_movies = filtered_movies.copy()
                filtered_movies["predicted_score"] = predictions
                top_recommendations = filtered_movies.sort_values("predicted_score", ascending=False).head(rec_count)
                
                # Complete progress bar
                progress_bar.progress(100)
                
                # Clear the progress bar
                progress_bar.empty()
                
                # Show results
                st.subheader(f"🎯 Your Top {rec_count} Movie Recommendations")
                st.markdown("""
                    <div class="info-box">
                        <p><strong>Match Score</strong> indicates how well each movie aligns with your taste profile, 
                        based on your rating history and similar users' preferences.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    for i, row in enumerate(top_recommendations.itertuples(), start=1):
                        score_percent = int(row.predicted_score * 100)
                        
                        # Extract year from title
                        year_match = row.title.strip().rfind("(")
                        if year_match > 0:
                            title_display = row.title[:year_match].strip()
                            year_display = row.title[year_match:]
                        else:
                            title_display = row.title
                            year_display = ""
                        
                        # Create card for each recommendation
                        st.markdown(f"""
                            <div class="recommendation-card">
                                <div class="movie-title">#{i}: {title_display} <span style="font-weight:400;color:#666;">{year_display}</span></div>
                                <div class="movie-genres">{row.genres.replace("|", " • ")}</div>
                                <div class="score-container">
                                    <div style="display:flex;justify-content:space-between;">
                                        <span>Match Score</span>
                                        <span><strong>{score_percent}%</strong></span>
                                    </div>
                                    <div class="score-bar">
                                        <div class="score-fill" style="width:{score_percent}%"></div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Display statistics about recommendations
                    st.subheader("Recommendation Insights")
                    
                    # Calculate genre distribution
                    genre_distribution = {}
                    for genres in top_recommendations["genres"]:
                        for genre in genres.split("|"):
                            if genre != "(no genres listed)":
                                genre_distribution[genre] = genre_distribution.get(genre, 0) + 1
                    
                    # Display top genres in recommendations
                    if genre_distribution:
                        st.markdown("#### Genre Distribution")
                        sorted_genres = sorted(genre_distribution.items(), key=lambda x: x[1], reverse=True)
                        for genre, count in sorted_genres[:5]:
                            percentage = (count / rec_count) * 100
                            st.markdown(f"- **{genre}**: {percentage:.0f}%")
                    
                    # Year distribution
                    st.markdown("#### Decade Distribution")
                    decades = {}
                    for year in top_recommendations["year"]:
                        if not pd.isna(year):
                            decade = int(year) // 10 * 10
                            decades[decade] = decades.get(decade, 0) + 1
                    
                    for decade, count in sorted(decades.items()):
                        percentage = (count / rec_count) * 100
                        st.markdown(f"- **{decade}s**: {percentage:.0f}%")
                    
                    # Average predicted score
                    avg_score = top_recommendations["predicted_score"].mean() * 100
                    st.markdown(f"#### Average Match Score: **{avg_score:.1f}%**")

            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                st.error("Please try again with different filters or contact support if the issue persists.")
else:
    # Show instructions when recommendations aren't generated yet
    st.markdown("""
        <div class="info-box" style="text-align: center; padding: 2rem;">
            <h3>Welcome to CineMatch!</h3>
            <p>Configure your recommendation settings in the sidebar, then click 'Generate Recommendations' to discover your personalized movie suggestions.</p>
        </div>
    """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("""
    <div style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e6e6e6; text-align: center; color: #666;">
        <p>CineMatch Movie Recommender | Powered by MovieLens Dataset | Neural Collaborative Filtering</p>
    </div>
""", unsafe_allow_html=True)