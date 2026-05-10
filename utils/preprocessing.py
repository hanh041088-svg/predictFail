def build_features(df):
    df["gpa"] = df.mean(axis=1)
    df["fail_count"] = (df < 5).sum(axis=1)
    df["std"] = df.std(axis=1)
    return df