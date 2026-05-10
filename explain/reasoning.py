def format_name(x):

    return (
        x.replace("_fail", "")
        .replace("_", " ")
        .title()
    )


def generate_reasoning(
    subject,
    shap_df,
    rules_df,
    result,
    student_row=None
):

    subject_name = format_name(
        subject
    )

    ignore_features = [
        "gpa",
        "std",
        "fail_count"
    ]

    # ======================
    # PASS
    # ======================
    if result == "PASS":

        return (
            f"Môn {subject_name} "
            f"có khả năng đạt yêu cầu "
            f"do kết quả học tập "
            f"tương đối ổn định."
        )

    causes = []

    # ======================
    # RULES FIRST
    # ======================
    if (
        rules_df is not None
        and not rules_df.empty
    ):

        best_rule = (
            rules_df
            .sort_values(
                "rule_strength",
                ascending=False
            )
            .iloc[0]
        )

        rule_feats = []

        for f in best_rule[
            "antecedent"
        ]:

            feat = f.replace(
                "_fail",
                ""
            )

            if (
                feat
                not in ignore_features
            ):

                if (
                    student_row is not None
                    and feat in student_row
                    and student_row[
                        feat
                    ] < 5
                ):
                    rule_feats.append(
                        feat
                    )

        causes.extend(
            rule_feats
        )

    # ======================
    # SHAP FALLBACK
    # ======================
    if len(causes) == 0:

        shap_feats = (
            shap_df
            .sort_values(
                "Impact",
                ascending=False
            )["Feature"]
            .tolist()
        )

        for feat in shap_feats:

            # bỏ feature engineered
            if (
                feat
                in ignore_features
            ):
                continue

            # phải dưới 5 mới là nguyên nhân
            if (
                student_row is not None
                and feat in student_row
                and student_row[
                    feat
                ] < 5
            ):

                causes.append(
                    feat
                )

            if len(causes) >= 3:
                break

    # ======================
    # FORMAT
    # ======================
    causes = [
        format_name(x)
        for x in causes
    ]

    if len(causes) == 0:

        return (
            f"Môn {subject_name} "
            f"có nguy cơ trượt "
            f"do kết quả học tập "
            f"chưa ổn định."
        )

    return (
        f"Môn {subject_name} "
        f"có nguy cơ trượt do "
        f"kết quả các môn "
        f"{', '.join(causes)} "
        f"chưa đạt yêu cầu."
    )