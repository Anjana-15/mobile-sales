import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit app title
st.title(" Mobile Sales Revenue Prediction for August 2024")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your 'mobile_sales.csv' file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show raw data
    st.subheader(" Uploaded Data Preview")
    st.dataframe(df.head())

    # ✅ Fix: Use correct column name 'Date' instead of 'Dates'
    df['date'] = pd.to_datetime(df['Date'])  # Removed dayfirst=True as the format is MM/DD/YYYY
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Drop unnecessary columns
    df = df.drop(columns=['TransactionID', 'Date', 'date'], errors='ignore')

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Split features and label
    X = df.drop(columns='TotalRevenue')
    y = df['TotalRevenue']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prepare future data (August 2025)
    future_dates = pd.date_range(start='2024-08-01', periods=30)
    future_df = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month,
        'day': future_dates.day
    })

    # Fill remaining columns with mode/mean
    for col in X.columns:
        if col in future_df.columns:
            continue
        future_df[col] = df[col].mode()[0] if df[col].dtype == 'uint8' else df[col].mean()

    # Reorder columns to match training data
    future_df = future_df[X.columns]
    future_scaled = scaler.transform(future_df)

    # Predict future revenue
    future_predictions = model.predict(future_scaled)

    # Create prediction output DataFrame
    prediction_output = pd.DataFrame({
        'Date': future_dates,
        'PredictedRevenue': future_predictions
    })

    # Show predictions
    st.subheader(" Predicted Revenue for August 2024")
    st.dataframe(prediction_output)

    # Plot results
    st.line_chart(prediction_output.set_index('Date'))

else:
    st.info("Please upload a CSV file to proceed.")