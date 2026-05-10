import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cf_predict(X_new, model_data, k=10):

    # ===== lấy feature đúng môn =====
    feats = model_data["features"]

    # ===== transform cùng scaler =====
    X_new_scaled = model_data["scaler_cf"].transform(X_new[feats])

    X_train_scaled = model_data["X_train_cf"]

    # ===== cosine similarity =====
    sim = cosine_similarity(X_new_scaled, X_train_scaled)
    sim = np.maximum(sim, 0)

    idx = np.argsort(sim[0])[::-1][:k]

    s = sim[0][idx]
    y = model_data["y_train_score"].iloc[idx].values

    score = np.sum(s * y) / (np.sum(s) + 1e-8)

    return np.clip(score, 0, 10)