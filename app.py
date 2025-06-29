import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

model = joblib.load('linear_model.pkl')
features = joblib.load('feature_names.pkl')

st.title("Smart Grid Load Forecasting")
st.subheader("Predict electricity load for a given date (06:00–15:00)")

forecast_date = st.date_input("Select forecast date", datetime(2025, 6, 29))

times = pd.date_range(f"{forecast_date} 06:00", f"{forecast_date} 15:00", freq='H')
df = pd.DataFrame({'time': times})
df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['weekday'] = df['time'].dt.weekday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)
df['lag_1'] = 0.95
df['lag_24'] = 0.90

X = df[features]
df['predicted_load'] = model.predict(X)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['time'], df['predicted_load'], color='orange', linewidth=2, label='Predicted Load')
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)
plt.grid(True)
plt.title(f"Forecast for {forecast_date}")
plt.xlabel("Time")
plt.ylabel("Load Power")
plt.legend()
st.pyplot(fig)

st.write("Forecasted Data")
st.dataframe(df[['time', 'predicted_load']])
