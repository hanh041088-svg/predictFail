import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE


def train_models(path):

    df = pd.read_csv(path, sep=";")
    df = df[df["student_id"] != "so_tin_chi"]

    # ===== CLEAN =====
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.fillna(df.mean(numeric_only=True))

    hk4 = df.columns[-6:]

    # ===== REMOVE GDTC =====
    ignore = [
        "giao_duc_the_chat_1",
        "giao_duc_the_chat_2",
        "giao_duc_the_chat_3",
        "ky_nang_mem"
    ]

    hk3 = [c for c in df.columns[1:-6] if c not in ignore]

    # ===== FEATURE ENGINEERING =====
    df["gpa"] = df[hk3].mean(axis=1)
    df["fail_count"] = (df[hk3] < 5).sum(axis=1)
    df["std"] = df[hk3].std(axis=1)

    base_features = hk3 + ["gpa", "fail_count", "std"]

    models = {}

    for target in hk4:

        y = (df[target] < 5).astype(int)
        X = df[base_features]

        X_train, _, y_train, _ = train_test_split(
            X, y, stratify=y, random_state=42
        )

        # ===== FEATURE SELECTION =====
        corr = X_train.corrwith(y_train).abs()
        selected = corr.sort_values(ascending=False).head(8).index.tolist()

        X_fs = X_train[selected]

        # ===== MODEL LR =====
        pipe_lr = Pipeline([
            ("imp", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("smote", BorderlineSMOTE()),
            ("clf", LogisticRegression(max_iter=1000))
        ])

        pipe_lr.fit(X_fs, y_train)
        prob_lr = pipe_lr.predict_proba(X_fs)[:, 1]

        # ===== MODEL RF =====
        pipe_rf = Pipeline([
            ("imp", SimpleImputer()),
            ("smote", BorderlineSMOTE()),
            ("clf", RandomForestClassifier(n_estimators=200))
        ])

        pipe_rf.fit(X_fs, y_train)
        prob_rf = pipe_rf.predict_proba(X_fs)[:, 1]

        # ===== TUNING THRESHOLD =====
        def best_f1(prob):
            best, best_th = 0, 0.5
            for th in np.arange(0.2, 0.8, 0.02):
                pred = (prob >= th).astype(int)
                f1 = f1_score(y_train, pred)
                if f1 > best:
                    best, best_th = f1, th
            return best, best_th

        f1_lr, th_lr = best_f1(prob_lr)
        f1_rf, th_rf = best_f1(prob_rf)

        if f1_rf > f1_lr:
            model = pipe_rf
            threshold = th_rf
            name = "RF"
        else:
            model = pipe_lr
            threshold = th_lr
            name = "LR"

        # ===== CF SPACE (CÙNG FEATURE + SCALE) =====
        scaler_cf = StandardScaler()
        X_cf = scaler_cf.fit_transform(X_fs)

        # ===== SHAP =====
        explainer = shap.Explainer(
            model.named_steps["clf"],
            X_fs
        )

        models[target] = {
            "pipe": model,
            "features": selected,
            "threshold": threshold,
            "model_name": name,

            # CF
            "X_train_cf": X_cf,
            "scaler_cf": scaler_cf,
            "y_train_score": df[target].loc[X_train.index],

            # explain
            "explainer": explainer
        }

    return models, base_features, hk4