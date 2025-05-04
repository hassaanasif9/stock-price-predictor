import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date, timedelta

# Title
st.title("📈 Stock Price Predictor")
st.write("Predict future stock prices using Linear Regression on historical data.")

# Choose data source
data_source = st.radio("Select Data Source:", ("Yahoo Finance", "Upload CSV (Kragle)"))

# Initialize df to avoid reference errors
df = None

# Load data
if data_source == "Yahoo Finance":
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL").strip().upper()
    start_date = st.date_input("Start Date", date.today() - timedelta(days=365))
    end_date = st.date_input("End Date", date.today())

    if st.button("Fetch Data"):
        try:
            # Convert dates to string format that yfinance expects
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data from Yahoo Finance
            df = yf.download(ticker, start=start_str, end=end_str, progress=False)
            
            # Check if data has been returned properly
            if df.empty:
                st.error(f"❌ No data returned for ticker {ticker}. Please check the ticker symbol and date range.")
                st.stop()

            # Reset index and ensure 'Date' is in datetime format
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone
            
            # Check if 'Close' column exists
            if 'Close' not in df.columns:
                st.error("❌ Data does not contain 'Close' column.")
                st.stop()

            st.success("✅ Data fetched successfully.")
            st.write(df.head())  # Display first few rows for verification
            
        except Exception as e:
            st.error(f"❌ Failed to fetch data: {e}")
            st.stop()

# Rest of your code remains the same...

elif data_source == "Upload CSV (Kragle)":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            else:
                st.error("❌ CSV must contain a 'Date' column.")
                st.stop()
            st.success("✅ File uploaded successfully.")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.stop()

# Check if df is defined and display data
if df is not None:
    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # Check if 'Date' and 'Close' columns exist
    if 'Date' in df.columns and 'Close' in df.columns:
        try:
            # Plot the data
            fig1 = px.line(df, x='Date', y='Close', title=f"{ticker if data_source=='Yahoo Finance' else 'Stock'} Closing Price Over Time")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating plot: {e}")
    else:
        st.error("❌ The data does not contain the required 'Date' or 'Close' columns for plotting.")

    # Feature Engineering
    st.subheader("⚙️ Feature Engineering")
    if len(df) > 1:
        df['Day'] = np.arange(len(df))  # Numeric index for regression
        X = df[['Day']]
        y = df['Close']

        # Check if there is enough data to split
        if len(df) > 1:
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Model Training
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            # Evaluation
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.metric("Mean Squared Error", f"{mse:.2f}")
            st.metric("R² Score", f"{r2:.4f}")

            # Actual vs Predicted chart
            st.subheader("📌 Actual vs Predicted Line Chart")
            result_df = pd.DataFrame({
                "Date": df['Date'].iloc[y_test.index],
                "Actual": y_test.values,
                "Predicted": y_pred
            })

            fig3 = px.line(result_df, x="Date", y=["Actual", "Predicted"], title="Actual vs Predicted Prices")
            st.plotly_chart(fig3, use_container_width=True)

            # Forecast next 7 days
            st.subheader("🔮 Forecast Next 7 Days")
            last_day = df['Day'].iloc[-1]
            future_days = np.array(range(last_day + 1, last_day + 8)).reshape(-1, 1)
            future_preds = model.predict(future_days)
            future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)

            forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})
            st.write(forecast_df)

            # Line chart for forecast
            fig2 = px.line(forecast_df, x='Date', y='Predicted Price', title="7-Day Stock Price Forecast")
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.error("❌ Not enough data to perform regression analysis. Please select a larger date range or ensure data is loaded correctly.")
    else:
        st.error("❌ Not enough data for feature engineering.")
else:
    st.warning("⚠️ Please fetch or upload stock data to continue.")
