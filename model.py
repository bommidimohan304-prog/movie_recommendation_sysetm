import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine as cosine_distance

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("TeluguMovies_dataset.csv")

df = df[['Movie', 'Genre', 'Rating', 'No.of.Ratings']].copy()

# -----------------------------
# CLEAN DATA
# -----------------------------
df = df.dropna(subset=['Movie', 'Genre', 'Rating', 'No.of.Ratings'])

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['No.of.Ratings'] = pd.to_numeric(df['No.of.Ratings'], errors='coerce')

df = df.dropna(subset=['Rating', 'No.of.Ratings'])

df = df[df['Rating'] > 0]
df = df[df['No.of.Ratings'] >= 0]

df = df.rename(columns={
    'Movie': 'movie_name',
    'Genre': 'genre',
    'Rating': 'rating',
    'No.of.Ratings': 'num_ratings'
})

# -----------------------------
# PROCESS GENRE
# -----------------------------
df['genre'] = df['genre'].apply(
    lambda x: [i.strip().lower() for i in str(x).split(',')]
)

# -----------------------------
# ENCODE GENRE
# -----------------------------
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genre'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# -----------------------------
# SCALE FEATURES
# -----------------------------
rating_scaler = MinMaxScaler()
pop_scaler = MinMaxScaler()

df['rating_scaled'] = rating_scaler.fit_transform(df[['rating']])

df['popularity_log'] = np.log1p(df['num_ratings'])
df['popularity_scaled'] = pop_scaler.fit_transform(df[['popularity_log']])

# -----------------------------
# FEATURE MATRIX
# -----------------------------
features = pd.concat([
    genre_df.reset_index(drop=True),
    (df['rating_scaled'] * 1.2).reset_index(drop=True),
    df['popularity_scaled'].reset_index(drop=True)
], axis=1)

features = features.fillna(0)

# -----------------------------
# TRAIN KNN
# -----------------------------
knn = NearestNeighbors(n_neighbors=20, metric='cosine')
knn.fit(features.values)

# -----------------------------
# DIVERSITY FUNCTION (MMR)
# -----------------------------
def diversify_results(candidate_indices, input_vector, top_n=5, lambda_param=0.7):

    selected = []
    candidate_vectors = features.iloc[candidate_indices].values

    while len(selected) < top_n and len(selected) < len(candidate_indices):

        best_idx = None
        best_score = -np.inf

        for i, idx in enumerate(candidate_indices):

            if idx in selected:
                continue

            sim_to_query = 1 - cosine_distance(input_vector, candidate_vectors[i])

            if len(selected) > 0:
                sim_to_selected = max([
                    1 - cosine_distance(candidate_vectors[i], features.iloc[j].values)
                    for j in selected
                ])
            else:
                sim_to_selected = 0

            score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)

    return selected

# -----------------------------
# USER-BASED BOOST
# -----------------------------
def adjust_with_user_history(results, username, users):

    if not users or username not in users or len(users[username]) == 0:
        return results

    history = users[username]

    preferred_genres = []
    for h in history:
        preferred_genres.extend(h['genre'].split(','))

    preferred_genres = [g.strip().lower() for g in preferred_genres]

    def boost(row):
        return len(set(row['genre']).intersection(preferred_genres))

    results['user_boost'] = results.apply(boost, axis=1)

    results = results.sort_values(by='user_boost', ascending=False)

    return results.drop(columns=['user_boost'])

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend_movies(input_genre, input_rating, top_n=5, username=None, users=None):

    input_genre = [i.strip().lower() for i in input_genre.split(',')]

    for g in input_genre:
        if g not in mlb.classes_:
            return []

    # Encode genre
    input_genre_encoded = mlb.transform([input_genre])

    # Scale rating
    input_rating_scaled = rating_scaler.transform(
        pd.DataFrame([[input_rating]], columns=['rating'])
    )[0][0]

    # Build input vector
    input_vector = np.concatenate([
        input_genre_encoded[0],
        [input_rating_scaled * 1.2],
        [0.5]
    ])

    # Get candidates
    distances, indices = knn.kneighbors([input_vector])

    candidate_indices = indices[0]

    # Apply diversity
    diverse_indices = diversify_results(
        candidate_indices,
        input_vector,
        top_n=top_n
    )

    results = df.iloc[diverse_indices].copy()

    # Add rating flexibility
    results['rating_diff'] = abs(results['rating'] - input_rating)

    results = results.sort_values(
        by=['rating_diff', 'num_ratings'],
        ascending=[True, False]
    )

    results = results.drop(columns=['rating_diff'])

    # Apply user learning
    results = adjust_with_user_history(results, username, users)

    return results[['movie_name', 'rating']].head(top_n).to_dict(orient='records')

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_model():

    distances, _ = knn.kneighbors(features)

    similarity_score = 1 - np.mean(distances)

    diversity_score = np.mean([
        cosine_distance(features.iloc[i], features.iloc[j])
        for i in range(len(features))
        for j in range(i+1, min(i+5, len(features)))
    ])

    return {
        "similarity_score": round(similarity_score, 3),
        "diversity_score": round(diversity_score, 3)
    }

# -----------------------------
# DEBUG
# -----------------------------
def check_data():
    print("NaN in features:", features.isnull().sum().sum())