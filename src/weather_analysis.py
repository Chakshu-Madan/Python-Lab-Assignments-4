import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RAW_DATA = Path("../data/raw_weather.csv")
CLEAN_DATA = Path("../data/cleaned_weather.csv")
IMG_DIR = Path("../images")
IMG_DIR.mkdir(exist_ok=True)

# -------------------------
# Task 1 – Load Dataset
# -------------------------
df = pd.read_csv(RAW_DATA)

print("HEAD:")
print(df.head())
print("\nINFO:")
print(df.info())
print("\nDESCRIBE:")
print(df.describe())

# -------------------------
# Task 2 – Cleaning
# -------------------------

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = df.dropna(subset=['date'])

num_cols = ['temperature', 'rainfall', 'humidity']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

df.to_csv(CLEAN_DATA, index=False)

# -------------------------
# Task 3 – Stats with NumPy
# -------------------------

daily_mean = np.mean(df['temperature'])
daily_min = np.min(df['temperature'])
daily_max = np.max(df['temperature'])
std_dev = np.std(df['temperature'])

print("\nTemperature Stats:")
print("Mean:", daily_mean)
print("Min:", daily_min)
print("Max:", daily_max)
print("Std Dev:", std_dev)

# Monthly aggregation
df['month'] = df['date'].dt.month
monthly_rainfall = df.groupby('month')['rainfall'].sum()

# Yearly aggregation (if year exists)
df['year'] = df['date'].dt.year
yearly_temp_stats = df.groupby('year')['temperature'].agg(['mean', 'min', 'max'])

# -------------------------
# Task 4 – Visualizations
# -------------------------

# 1. Line chart – daily temperature
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['temperature'])
plt.title("Daily Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.savefig(IMG_DIR/"daily_temperature.png")
plt.close()

# 2. Bar chart – monthly rainfall
plt.figure(figsize=(10,5))
monthly_rainfall.plot(kind='bar')
plt.title("Monthly Rainfall Total")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.savefig(IMG_DIR/"monthly_rainfall.png")
plt.close()

# 3. Scatter – humidity vs temperature
plt.figure(figsize=(8,5))
plt.scatter(df['temperature'], df['humidity'])
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.savefig(IMG_DIR/"humidity_vs_temp.png")
plt.close()

# 4. Combined plot
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(df['date'], df['temperature'], color='red')
plt.title("Daily Temperature Trend")

plt.subplot(2,1,2)
plt.bar(monthly_rainfall.index, monthly_rainfall.values, color='blue')
plt.title("Monthly Rainfall Totals")
plt.tight_layout()
plt.savefig(IMG_DIR/"combined_plots.png")
plt.close()

print("\nAll plots saved in /images folder.")
