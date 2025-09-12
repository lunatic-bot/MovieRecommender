# Movie Recommendation System

## Overview

This project builds a **Movie Recommendation System** using the **MovieLens 100k dataset**.
It explores both **classical ML (Matrix Factorization with SVD)** and a **Deep Learning approach (Neural Collaborative Filtering)** to recommend movies to users.

Finally, the system is deployed with a **Streamlit web app** where users can get personalized recommendations.

---

## Features

- Data preprocessing (user, movie, ratings).
- Collaborative Filtering with SVD (baseline).
- Neural Collaborative Filtering (embeddings + MLP).
- Generate **Top-N recommendations** for a given user or movie.
- Streamlit UI for interactive movie suggestions.

---

## Project Structure

```
movie-recommender/
│── data/                # MovieLens dataset
│── notebooks/           # Exploration and experiments
│── src/
│   ├── preprocessing.py # Data loading & encoding
│   ├── baseline.py      # SVD / Surprise implementation
│   ├── ncf.py           # Neural Collaborative Filtering
│   ├── recommend.py     # Recommendation logic
│   └── utils.py
│── app/
│   └── streamlit_app.py # Web UI
│── models/              # Saved models
│── README.md
```

---

## Dataset

- **MovieLens 100k**: 100,000 ratings (1–5) from 943 users on 1682 movies.
- Download: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/).

---

## Workflow

1. **Data Preprocessing**

   - Encode `userId` and `movieId` into numeric indices.
   - Train/test split (80/20).

2. **Baseline Model (SVD)**

   - Matrix factorization using `Surprise` or `TruncatedSVD`.

3. **Neural Collaborative Filtering**

   - Learn embeddings for users and movies.
   - Train a small MLP to predict ratings.

4. **Recommendation Generation**

   - Predict scores for unseen movies.
   - Sort and return top-k recommendations.

5. **Deployment**

   - Streamlit app to input a user ID or movie name.
   - Display top movie suggestions (optionally with posters via TMDb API).

---

## 🛠 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, Surprise
- PyTorch / TensorFlow (for NCF)
- Streamlit (deployment)

---

## Future Improvements

- Add hybrid recommendations (content + collaborative).
- Integrate TMDb API for posters and movie details.
- Deploy online (Heroku / Render / HuggingFace Spaces).
