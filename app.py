import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Data Analytics Assistant", layout="wide")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

ROLES = {
    "expert": "Technical, precise and direct responses",
    "didactic": "Step-by-step explanations",
    "executive": "High-level business-oriented summaries"
}

# =========================
# SESSION
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if "df" not in st.session_state:
    st.session_state.df = None

if "data_profile" not in st.session_state:
    st.session_state.data_profile = None

# =========================
# FUNCTIONS
# =========================
def generate_data_profile(df):
    return {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isnull().sum().to_dict()
    }

def detect_intent(query):
    query = query.lower()
    if "code" in query and "explain" in query:
        return "both"
    elif "code" in query:
        return "code"
    elif "explain" in query or "why" in query:
        return "explanation"
    else:
        return "both"

def build_prompt(data_profile, user_query, role, intent):

    dataset_context = json.dumps(data_profile, indent=2) if data_profile else "No dataset provided"

    system_prompt = f"""
You are an expert in data science.

OUTPUT JSON:
{{
    "code": "Python code or empty string",
    "explanation": "Spanish explanation or empty string"
}}

RULES:
- Detect if user wants code, explanation, or both
- If no dataset, generate generic examples
- Explanation MUST be in Spanish
- Code must be executable

ROLE: {role}
INTENT: {intent}

DATASET:
{dataset_context}
"""

    return system_prompt, user_query

def query_openai(system_prompt, user_prompt):
    response = client.responses.create(
        model="gpt-5.4-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.output_text

def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])

# =========================
# UI
# =========================
st.title("Data Science Assistant")

uploaded_file = st.file_uploader("Upload dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.session_state.data_profile = generate_data_profile(df)

    st.subheader("Preview")
    st.dataframe(df.head())

    # =========================
    # SUMMARY (PARALLEL)
    # =========================
    with st.expander("Dataset Summary"):

        col1, col2 = st.columns(2)

        with col1:
            st.write("Info")
            info_df = pd.DataFrame({
                "column": df.columns,
                "dtype": df.dtypes.astype(str),
                "nulls": df.isnull().sum()
            })
            st.dataframe(info_df)

        with col2:
            st.write("Describe")
            st.dataframe(df.describe())

    # =========================
    # VISUALS
    # =========================
    with st.expander("Quick Visualizations"):

        cols = df.columns.tolist()

        x = st.selectbox("X variable", cols)
        y = st.selectbox("Y variable (optional)", [None] + cols)

        plot_type = st.selectbox("Plot type", [
            "bar", "line", "box", "scatter", "hist_kde"
        ])

        fig, ax = plt.subplots()

        if plot_type == "bar":
            sns.barplot(x=df[x], y=df[y] if y else None, ax=ax)
        elif plot_type == "line":
            sns.lineplot(x=df[x], y=df[y] if y else None, ax=ax)
        elif plot_type == "box":
            sns.boxplot(x=df[x], y=df[y] if y else None, ax=ax)
        elif plot_type == "scatter" and y:
            sns.scatterplot(x=df[x], y=df[y], ax=ax)
        elif plot_type == "hist_kde":
            sns.histplot(df[x], kde=True, ax=ax)

        st.pyplot(fig)

# =========================
# CONTROLS
# =========================
role = st.selectbox("Select assistant role", list(ROLES.keys()))
user_query = st.text_area("Ask your question")

# =========================
# EXECUTION
# =========================
if st.button("Generate"):

    if not user_query.strip():
        st.warning("Enter a valid query")
    else:
        intent = detect_intent(user_query)

        system_prompt, user_prompt = build_prompt(
            st.session_state.data_profile,
            user_query,
            role,
            intent
        )

        raw = query_openai(system_prompt, user_prompt)

        try:
            parsed = safe_json_parse(raw)
            code = parsed.get("code", "")
            explanation = parsed.get("explanation", "")

            col1, col2 = st.columns(2)

            with col1:
                if code:
                    st.subheader("Code")
                    st.code(code)
                    st.download_button("Download .py", code, "code.py")

            with col2:
                if explanation:
                    st.subheader("Explanation")
                    st.write(explanation)

            st.session_state.history.append({
                "query": user_query,
                "code": code,
                "explanation": explanation
            })

        except:
            st.error("Parsing error")
            st.text(raw)

# =========================
# HISTORY
# =========================
if st.session_state.history:
    st.subheader("History")

    for item in reversed(st.session_state.history):
        with st.expander(item["query"]):
            st.write(item["explanation"])
            st.code(item["code"])