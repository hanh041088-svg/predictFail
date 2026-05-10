import json
import pandas as pd


# ==================================
# LOAD RULES
# ==================================
def load_rules(path="rules/rules.json"):

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ===== normalize =====
    df["antecedent"] = df["antecedent"].apply(
        lambda x: list(x)
        if isinstance(x, list)
        else []
    )

    df["consequent"] = df["consequent"].apply(
        lambda x: list(x)
        if isinstance(x, list)
        else []
    )

    # ===== normalize target =====
    if "target" in df.columns:
        df["target"] = (
            df["target"]
            .astype(str)
            .str.replace(
                "_fail",
                "",
                regex=False
            )
        )

    # ===== rule strength fallback =====
    if "rule_strength" not in df.columns:

        if (
            "confidence" in df.columns
            and "lift" in df.columns
        ):
            df["rule_strength"] = (
                df["confidence"]
                * df["lift"]
            )
        else:
            df["rule_strength"] = 0

    # ===== sort =====
    df = df.sort_values(
        by=["rule_strength", "confidence"],
        ascending=False
    )

    return df


# ==================================
# MATCH RULES
# ==================================
def match_rules(
    student_row,
    rules_df,
    subject
):

    if rules_df is None or rules_df.empty:
        return pd.DataFrame()

    subject = (
        subject
        .replace("_fail", "")
        .strip()
    )

    # ===== lấy đúng môn =====
    sub_rules = rules_df[
        rules_df["target"] == subject
    ].copy()

    if sub_rules.empty:
        return pd.DataFrame()

    matched = []

    for _, rule in sub_rules.iterrows():

        antecedent = rule["antecedent"]

        ok = True

        for item in antecedent:

            feature = item.replace(
                "_fail",
                ""
            )

            # không tồn tại môn
            if feature not in student_row.index:
                ok = False
                break

            # không fail => rule không khớp
            if student_row[feature] >= 5:
                ok = False
                break

        if ok:
            matched.append(rule)

    if len(matched) == 0:
        return pd.DataFrame()

    matched = pd.DataFrame(matched)

    matched = matched.sort_values(
        by=[
            "rule_strength",
            "confidence",
            "lift"
        ],
        ascending=False
    )

    return matched.head(3)


# ==================================
# RULE TO TEXT
# ==================================
def rules_to_text(rules_df):

    if (
        rules_df is None
        or rules_df.empty
    ):
        return [
            "Không có luật phù hợp"
        ]

    texts = []

    for _, r in rules_df.iterrows():

        antecedent = [
            x.replace("_fail", "")
            .replace("_", " ")
            .title()
            for x in r["antecedent"]
        ]

        consequent = [
            x.replace("_fail", "")
            .replace("_", " ")
            .title()
            for x in r["consequent"]
        ]

        confidence = round(
            r.get("confidence", 0) * 100,
            1
        )

        lift = round(
            r.get("lift", 0),
            2
        )

        support = round(
            r.get("support", 0) * 100,
            1
        )

        text = (
            f"Nếu yếu "
            f"{', '.join(antecedent)} "
            f"→ nguy cơ gặp khó khăn ở "
            f"{', '.join(consequent)} "
            f"(confidence={confidence}%, "
            f"lift={lift}, "
            f"support={support}%)"
        )

        texts.append(text)

    return texts