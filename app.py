
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle

app = FastAPI()

# Load model and genre data
model = tf.keras.models.load_model("models/hybrid_ncf_model.h5")
with open("models/movie_idx_to_genre.pkl", "rb") as f:
    movie_idx_to_genre = pickle.load(f)

class RecommendationRequest(BaseModel):
    user_idx: int
    movie_idx: int

@app.post("/predict/")
def predict_rating(req: RecommendationRequest):
    user_input = np.array([req.user_idx])
    movie_input = np.array([req.movie_idx])
    genre_input = np.array([movie_idx_to_genre.get(req.movie_idx, np.zeros(18))])

    pred = model.predict({
        "user_input": user_input,
        "item_input": movie_input,
        "genre_input": genre_input
    }, verbose=0)[0][0]

    return {"score": float(pred)}
