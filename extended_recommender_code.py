
# --- Extension: Add Genres as Metadata ---

# 1. One-hot encode genres for hybrid modeling
from sklearn.preprocessing import MultiLabelBinarizer

# Split genre strings and one-hot encode
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
genre_features = pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_)
genre_features['movieId'] = movies['movieId']

# Merge genre features into ratings
ratings = ratings.merge(genre_features, how='left', on='movieId')

# Optional: reduce dimensionality of genre one-hot vectors if desired

# --- Optional Hybrid Model using Metadata ---
# You can include these features as an additional input branch or concatenate to item embeddings

# --- Extension: Session-Based or Sequential Modeling with RNNs/Transformers ---

# Sort and group interactions by user and timestamp
ratings = ratings.sort_values(['userId', 'timestamp'])

# Create interaction sequences for each user (e.g., sliding window of movieId history)
# This part can be used to train RNNs or Transformer models on sequences
user_sequences = ratings.groupby('userId')['movieId'].apply(list)

# Example: Prepare sequences of length N with next-item prediction
# Useful for sequential models like GRU4Rec, SASRec, etc.

# You can use torch.nn.TransformerEncoder or nn.GRU with padding + masking

# --- Evaluation Metrics for Top-K Ranking ---

import numpy as np

def recall_at_k(actual, predicted, k):
    predicted = predicted[:k]
    return len(set(actual) & set(predicted)) / len(actual)

def ndcg_at_k(actual, predicted, k):
    predicted = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(actual, predicted, k):
    predicted = predicted[:k]
    return 1.0 if len(set(actual) & set(predicted)) > 0 else 0.0
