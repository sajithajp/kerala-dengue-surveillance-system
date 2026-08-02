import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

# Load Cleaned Data
df = pd.read_csv('D:/Article/Dengue/dengue-forecasting-app/dengue_monthly_time_series.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Train SARIMAX Model (Seasonal Order for Monthly Monsoon Spikes)
model = SARIMAX(df['Dengue_Cases'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Save Model
joblib.dump(results, 'dengue_sarima_model.pkl')
print("✅ SARIMA Model trained and saved as 'dengue_sarima_model.pkl'!")
