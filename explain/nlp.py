def format_name(name):
    return name.replace("_", " ").title()


def generate_explanation(subject, shap_df, result):

    subject = format_name(subject)

    if result == "PASS":
        return f"Môn {subject} có khả năng đạt yêu cầu do kết quả học tập ổn định."

    weak = shap_df[shap_df["Impact"] > 0]["Feature"].head(2).tolist()
    weak = [format_name(w) for w in weak]

    if len(weak) == 1:
        return f"Môn {subject} có nguy cơ trượt do kết quả môn {weak[0]} chưa tốt."

    return (
        f"Môn {subject} có nguy cơ trượt do kết quả các môn "
        f"{weak[0]} và {weak[1]} chưa đạt yêu cầu."
    )

def generate_advice(shap_df):

    weak = shap_df[shap_df["Impact"] > 0]["Feature"].head(2).tolist()
    weak = [format_name(w) for w in weak]

    if not weak:
        return "Tiếp tục duy trì kết quả học tập."

    return f"Nên tập trung cải thiện các môn {', '.join(weak)} và luyện tập thường xuyên."