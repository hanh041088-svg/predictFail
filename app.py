import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from sklearn.metrics.pairwise import cosine_similarity

from models.train_model import train_models
from utils.preprocessing import build_features

from explain import (
    get_shap_df,
    load_rules,
    match_rules,
    rules_to_text,
    generate_reasoning
)


# ==================================
# CF PREDICT
# ==================================
def cf_predict(X_new, m):

    feats = m["features"]

    X_new_scaled = (
        m["scaler_cf"]
        .transform(X_new[feats])
    )

    X_train = m["X_train_cf"]

    sim = cosine_similarity(
        X_new_scaled,
        X_train
    )

    sim = np.maximum(sim, 0)

    idx = np.argsort(
        sim[0]
    )[::-1][:10]

    s = sim[0][idx]

    y = (
        m["y_train_score"]
        .iloc[idx]
        .values
    )

    score = np.sum(
        s * y
    ) / (np.sum(s) + 1e-8)

    return np.clip(score, 0, 10)


# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Student Risk Prediction",
    layout="wide"
)

st.title(
    "🎓 Student Risk Prediction System"
)


# ==================================
# LOAD MODEL
# ==================================
@st.cache_resource
def load_all():

    models, features, hk4 = (
        train_models(
            "data/clean_hk4-ktpm.csv"
        )
    )

    rules = load_rules(
        "rules/rules.json"
    )

    return (
        models,
        features,
        hk4,
        rules
    )


models, features, hk4, rules_df = (
    load_all()
)


# ==================================
# INPUT
# ==================================
st.subheader("📘 Nhập điểm HK3")

inputs = {}

cols = st.columns(3)

for i, f in enumerate(features[:-3]):

    with cols[i % 3]:

        inputs[f] = st.number_input(
            label=f,
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )


# ==================================
# BUTTON
# ==================================
if st.button("🔍 Dự đoán"):

    # =====================
    # BUILD INPUT
    # =====================
    df_new = pd.DataFrame(
        [inputs]
    )

    df_new = build_features(
        df_new
    )

    results = []

    # =====================
    # LOOP SUBJECT
    # =====================
    for subject in hk4:

        st.markdown("---")

        subject_name = (
            subject
            .replace("_", " ")
            .title()
        )

        st.subheader(
            f"📘 {subject_name}"
        )

        m = models[subject]

        X_new = (
            df_new[
                m["features"]
            ]
        )

        # =====================
        # CLASSIFICATION
        # =====================
        prob = (
            m["pipe"]
            .predict_proba(X_new)[0][1]
        )

        # =====================
        # CF
        # =====================
        cf_score = cf_predict(
            df_new,
            m
        )

        # =====================
        # HYBRID
        # =====================
        final_score = (
            0.7 * cf_score
            +
            0.3 * (
                10 * (1 - prob)
            )
        )

        result = (
            "FAIL"
            if final_score < 5
            else "PASS"
        )

        # =====================
        # SUMMARY
        # =====================
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "CF Score",
                round(cf_score, 2)
            )

        with c2:
            st.metric(
                "Fail Probability",
                round(prob, 2)
            )

        with c3:
            st.metric(
                "Final Score",
                round(
                    final_score,
                    2
                )
            )

        st.write(
            f"### {'❌ FAIL' if result=='FAIL' else '✅ PASS'}"
        )

        # =====================
        # SHAP
        # =====================
        st.subheader(
            "🧠 SHAP"
        )

        shap_df = get_shap_df(
            m["explainer"],
            X_new
        )

        st.dataframe(
            shap_df.head(5),
            use_container_width=True
        )

        # =====================
        # RULES
        # =====================
        matched = pd.DataFrame()
        # chỉ dùng rules nếu nguy cơ FAIL
        if result == "FAIL":
            matched = match_rules(df_new.iloc[0],rules_df,subject)
        st.subheader("📌 Association Rules")
        if result == "FAIL":
            if matched.empty:
                st.info("Không có luật phù hợp")
            else:
                for txt in rules_to_text(matched):
                    st.error(txt)
        else:
            st.success("Không phát hiện mẫu rủi ro đáng kể "
        "từ dữ liệu lịch sử.")

        
        # =====================
        # REASONING
        # =====================
        st.subheader(
            "📝 Giải thích"
        )

        explanation = (generate_reasoning(
            subject,
            shap_df,
            matched,
            result,
            df_new.iloc[0]
            )
        )

        st.info(explanation)

        results.append([
            subject_name,
            round(
                final_score,
                2
            ),
            result
        ])

    # =====================
    # FINAL TABLE
    # =====================
    st.markdown("---")

    st.subheader(
        "📊 Tổng kết"
    )

    result_df = pd.DataFrame(
        results,
        columns=[
            "Subject",
            "Score",
            "Result"
        ]
    )

    st.dataframe(
        result_df,
        use_container_width=True
    )

    fail_count = (
        result_df["Result"]
        .eq("FAIL")
        .sum()
    )

    st.error(
        f"❌ Số môn nguy cơ trượt: "
        f"{fail_count}/"
        f"{len(hk4)}"
    )