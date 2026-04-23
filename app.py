import streamlit as st
import pandas as pd
import json
from openai import OpenAI

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Data Analytics Assistant", layout="wide")

# =========================
# LOAD API KEY FROM SECRETS
# =========================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================
# CONSTANTS
# =========================
SECTIONS = {
    "eda": "Exploratory Data Analysis",
    "data_cleaning": "Data Cleaning",
    "feature_engineering": "Feature Engineering",
    "feature_selection": "Feature Selection",
    "modeling": "Predictive Modeling",
    "hyperparameter_optimization": "Hyperparameter Optimization",
    "visualization": "Data Visualization",
    "statistics": "Statistical Analysis"
}

ROLES = {
    "expert": "Technical, precise and direct responses",
    "didactic": "Step-by-step explanations",
    "executive": "High-level business-oriented summaries"
}

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if "data_profile" not in st.session_state:
    st.session_state.data_profile = None

# =========================
# FUNCTIONS
# =========================
def generate_data_profile(df):
    try:
        profile = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "nulls": df.isnull().sum().to_dict(),
            "describe": df.describe(include='all').fillna("").to_dict(),
            "sample": df.head(5).to_dict()
        }
        return profile
    except Exception as e:
        return {"error": str(e)}

def build_prompt(data_profile, user_query, section, role):
    system_prompt = f"""
You are an expert in data science and advanced analytics.

CONSTRAINTS:
- Only respond to analytics, statistics, or data science topics.
- If out of scope, respond exactly:
"This tool is exclusively focused on data analytics."

MANDATORY OUTPUT FORMAT (valid JSON):
{{
    "code": "Complete Python code with imports",
    "explanation": "Professional explanation in Spanish"
}}

CODE RULES:
- Must be executable
- Include imports
- Do not invent columns
- Use pandas, numpy, sklearn, matplotlib or seaborn
- No text outside code in 'code'

ASSISTANT ROLE: {role}
TASK TYPE: {section}

DATASET CONTEXT:
{json.dumps(data_profile, indent=2)}
"""

    user_prompt = f"""
User request:
{user_query}
"""

    return system_prompt, user_prompt

def query_openai(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="gpt-5.3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# =========================
# UI
# =========================
st.title("Data Science & Analytics Assistant")

uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.session_state.data_profile = generate_data_profile(df)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        with st.expander("Dataset Summary"):
            st.json(st.session_state.data_profile)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

section = st.selectbox(
    "Select task type",
    options=list(SECTIONS.keys()),
    format_func=lambda x: SECTIONS[x]
)

role = st.selectbox(
    "Select assistant role",
    options=list(ROLES.keys()),
    format_func=lambda x: ROLES[x]
)

user_query = st.text_area("Describe your request")

if st.button("Generate response"):

    if st.session_state.data_profile is None:
        st.warning("Upload a dataset first")
    elif not user_query.strip():
        st.warning("Enter a valid request")
    else:
        with st.spinner("Generating response..."):

            system_prompt, user_prompt = build_prompt(
                st.session_state.data_profile,
                user_query,
                section,
                role
            )

            raw_response = query_openai(system_prompt, user_prompt)

            try:
                parsed = json.loads(raw_response)

                code = parsed.get("code", "")
                explanation = parsed.get("explanation", "")

                st.subheader("Generated Code")
                st.code(code, language="python")

                st.download_button(
                    label="Download code (.py)",
                    data=code,
                    file_name="generated_code.py",
                    mime="text/x-python"
                )

                st.subheader("Explanation (Spanish)")
                st.write(explanation)

                st.session_state.history.append({
                    "query": user_query,
                    "section": section,
                    "role": role,
                    "code": code,
                    "explanation": explanation
                })

            except Exception:
                st.error("Error parsing model response")
                st.text(raw_response)

# =========================
# HISTORY
# =========================
if st.session_state.history:
    st.subheader("Query History")

    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Query {i}: {item['query']}"):
            st.write(f"Section: {SECTIONS[item['section']]}")
            st.write(f"Role: {ROLES[item['role']]}")
            st.code(item["code"], language="python")
            st.write(item["explanation"])