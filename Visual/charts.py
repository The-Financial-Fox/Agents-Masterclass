import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="📊 Advanced Financial Visualizer", page_icon="📈", layout="wide")
st.title("📊 Advanced Financial Data Visualizer")
st.write("Upload an Excel/CSV file, choose chart types, or let AI suggest visualizations.")

# File Upload
uploaded_file = st.file_uploader("📤 Upload Excel or CSV File", type=["xlsx", "csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.success("✅ File Uploaded and Loaded")

    st.subheader("📁 Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

    st.subheader("📈 Choose Advanced Chart Type")

    chart_type = st.selectbox("🧠 Select Visualization Type", [
        "Line Chart", "Bar Chart", "Area Chart", "Scatter Plot",
        "Box Plot", "Histogram", "Correlation Heatmap", "Dual Axis Line Chart",
        "Time Series (Plotly)"
    ])

    if chart_type == "Correlation Heatmap":
        st.subheader("📊 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    elif chart_type == "Dual Axis Line Chart":
        col1 = st.selectbox("Primary Y-Axis", numeric_cols, key="dual_y1")
        col2 = st.selectbox("Secondary Y-Axis", numeric_cols, key="dual_y2")
        x_axis = st.selectbox("X-Axis", df.columns, key="dual_x")
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax2 = ax1.twinx()
        ax1.plot(df[x_axis], df[col1], color='blue')
        ax2.plot(df[x_axis], df[col2], color='green')

        ax1.set_ylabel(col1, color='blue')
        ax2.set_ylabel(col2, color='green')
        ax1.set_xlabel(x_axis)

        st.pyplot(fig)

    elif chart_type == "Time Series (Plotly)":
        time_col = st.selectbox("Time Column", df.columns, key="ts_time")
        y_col = st.selectbox("Value Column", numeric_cols, key="ts_y")
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        fig = px.line(df, x=time_col, y=y_col, title=f"{y_col} over Time")
        st.plotly_chart(fig)

    else:
        x_col = st.selectbox("X-Axis", df.columns, key="basic_x")
        y_col = st.selectbox("Y-Axis", numeric_cols, key="basic_y")

        fig, ax = plt.subplots(figsize=(10, 5))
        if chart_type == "Line Chart":
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == "Bar Chart":
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == "Area Chart":
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.fill_between(df[x_col], df[y_col], alpha=0.3)
        elif chart_type == "Scatter Plot":
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == "Histogram":
            sns.histplot(data=df, x=y_col, bins=20, ax=ax)
        elif chart_type == "Box Plot":
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)

        st.pyplot(fig)

    # AI Insight
    st.subheader("🤖 AI Commentary on Trends")
    sample_data = df.head(100).to_json(orient="records")

    prompt = f"""
    You are a data analyst. Based on the following financial dataset sample, suggest:
    - What are the best chart types to use?
    - What are the key trends or anomalies?
    - Write 3 executive-level insights from this data.

    Sample Data:
    {sample_data}
    """

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a financial data visualization and storytelling expert."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )

    st.markdown(response.choices[0].message.content)
