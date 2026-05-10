import pandas as pd
import numpy as np

def get_shap_df(explainer, X_new):

    shap_values = explainer(X_new)

    values = shap_values.values

    # ===== XỬ LÝ MỌI TRƯỜNG HỢP =====
    if isinstance(values, list):
        values = values[1]  # class FAIL

    values = np.array(values)

    # nếu 3D → lấy class 1
    if len(values.shape) == 3:
        values = values[:, :, 1]

    # nếu vẫn còn 2D → lấy dòng đầu
    if len(values.shape) == 2:
        values = values[0]

    # ép về 1D chắc chắn
    values = values.flatten()

    df = pd.DataFrame({
        "Feature": X_new.columns,
        "Impact": values
    })

    return df.sort_values(by="Impact", ascending=False)

# =========================
# LẤY FEATURE GÂY FAIL
# =========================
def get_risk_features(shap_df, top_k=3):

    df = shap_df.copy()

    # chỉ lấy feature làm tăng nguy cơ
    df = df[df["Impact"] > 0]

    if df.empty:
        return []

    return df.head(top_k)["Feature"].tolist()


# =========================
# CHUẨN HÓA SHAP (CHO GIẢI THÍCH)
# =========================
def normalize_shap(shap_df):

    df = shap_df.copy()

    if df.empty:
        return df

    min_v = df["Impact"].min()
    max_v = df["Impact"].max()

    if max_v - min_v < 1e-6:
        df["Impact_norm"] = 0.5
    else:
        df["Impact_norm"] = (df["Impact"] - min_v) / (max_v - min_v)

    return df


# =========================
# TOP SHAP (DẠNG TEXT)
# =========================
def shap_to_text(shap_df, top_k=3):

    feats = get_risk_features(shap_df, top_k)

    return [f.replace("_", " ") for f in feats]