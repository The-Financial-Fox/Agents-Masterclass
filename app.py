#This is where you add your code
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from prophet import Prophet
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="📈 AI Forecasting Agent", page_icon="📊", layout="wide")

st.title("📈 Revenue Forecasting using Prophet")
st.write("Upload an Excel file with **Date** and **Revenue** columns.")

# File upload
uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()  # Remove extra whitespace
        if "Date" not in df.columns or "Revenue" not in df.columns:
            st.error("❌ Excel must have 'Date' and 'Revenue' columns.")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df[["Date", "Revenue"]].dropna()

        # Rename for Prophet
        prophet_df = df.rename(columns={"Date": "ds", "Revenue": "y"})

        # Prophet Forecasting
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        # Plot
        st.subheader("🔮 Forecast Plot")
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Forecast Table
        st.subheader("📊 Forecast Table (Next 12 Months)")
        forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12)
        st.dataframe(forecast_result.rename(columns={
            "ds": "Date",
            "yhat": "Forecasted Revenue",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound"
        }))

        # AI Commentary
        st.subheader("🧠 AI Forecast Commentary")

        # Prepare prompt
        data_json = forecast_result.to_json(orient="records", date_format="iso")
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        You are an FP&A forecasting expert.
        Please analyze the following forecasted revenue data and provide:
        - Key trends in revenue
        - Any risks or volatility
        - A brief 3-bullet summary a CFO would care about

        Forecast Data:
        {data_json}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a forecasting and FP&A expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        st.markdown(response.choices[0].message.content)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
