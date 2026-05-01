import streamlit as st
import pandas as pd
import os
import io
from dotenv import load_dotenv
from google import genai
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
import numpy as np

# =========================
# 🔑 LOAD ENV
# =========================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API Key not found. Check .env file")

# Gemini client
client = genai.Client(api_key=api_key)

# =========================
# 🧠 SIMPLE EMBEDDING (RAG)
# =========================
def create_embeddings(text_list):
    return np.array([
        [hash(text) % 1000000 / 1000000 for _ in range(10)]
        for text in text_list
    ]).astype("float32")

# =========================
# 🚀 UI START
# =========================
st.title("📊 InsightForge AI")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:

    # =========================
    # 📂 READ FILE
    # =========================
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    # =========================
    # 📊 EDA
    # =========================
    st.write("### Basic Info")
    st.write(df.describe())

    st.write("### Missing Null Values")
    st.write(df.isnull().sum())

    st.write("### Table Info and Data Types")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # =========================
    # 🧠 RAG SETUP
    # =========================
    df["combined"] = df.astype(str).agg(" | ".join, axis=1)
    documents = df["combined"].tolist()

    embeddings = create_embeddings(documents)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    def retrieve(query, k=3):
        query_embedding = create_embeddings([query])
        distances, indices = index.search(query_embedding, k)
        results = [documents[i] for i in indices[0]]
        return "\n".join(results)

    # =========================
    # 📄 DATASET SUMMARY (RAG)
    # =========================
    st.write("### 📄 Dataset Summary (AI Generated)")

    if st.button("Generate Summary"):

        context = "\n".join(documents[:5])  # small sample for summary

        summary_prompt = """
        Assume that you are a Data Analyst. you need to analyze the dataset and 
        provide a comprehensive summary. Focus on key statistics, trends, and any
        notable insights that can be derived from the data.
        if there are any missing values, mention them and suggest how to handle them.
        Suggest five potential business recommendations based on the dataset.

        Dataset Sample:
        {context}

        Provide a clear summary of this dataset in around 200-250 words.
        """

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=summary_prompt
        )

        st.write(response.text)

    # =========================
    # 🤖 CHAT WITH RAG
    # =========================
    query = st.text_input("Ask something about your data")

    if query:

        context = retrieve(query)

        prompt = f"""
        You are a Data Analyst.

        Relevant Data:
        {context}

        Dataset columns: {list(df.columns)}

        User question: {query}
        
        Give:
        Based on the following dataset context, answer the user's query.
        If the answer is not present in the context, say "The answer is not
        present in the Dataset. 
        
        The answer should be in form of bullet points and make sure it is well-structured
        and easy to read.
        
        The summary should be either in form of paragraph or in form of bullet points
        Based on the context provided. Do not use both formats together.
        
        The summary should not exceed 200 words. Focus on key insights and actionable 
        recommendations.
        """

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        st.write("### 🤖 Answer")
        st.write(response.text)

    # =========================
    # 📊 CUSTOM CHART BUILDER
    # =========================
    st.write("### 📊 Custom Chart Builder")

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Line", "Bar", "Scatter", "Histogram"]
    )

    x_col = st.selectbox("Select X-axis", df.columns)
    y_col = st.selectbox("Select Y-axis", df.columns)

    color = st.color_picker("Pick a color", "#1f77b4")

    if "charts" not in st.session_state:
        st.session_state.charts = []

    if st.button("Add Chart"):
        st.session_state.charts.append((chart_type, x_col, y_col, color))

    # =========================
    # 📈 MULTIPLE CHARTS
    # =========================
    for i, (ctype, x, y, c) in enumerate(st.session_state.charts):
        st.write(f"### Chart {i+1}")

        plt.figure()

        try:
            if ctype == "Line":
                plt.plot(df[x], df[y], color=c)

            elif ctype == "Bar":
                plt.bar(df[x], df[y], color=c)

            elif ctype == "Scatter":
                plt.scatter(df[x], df[y], color = c)

            elif ctype == "Histogram":
                sns.histplot(df[x], color = c, kde = True)

            plt.xlabel(x)
            plt.ylabel(y)

            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error generating chart: {e}")