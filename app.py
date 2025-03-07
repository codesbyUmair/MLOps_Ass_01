import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)
num_samples = 500

# Generate synthetic traffic data
time_of_day = np.random.randint(0, 24, num_samples)
temperature = np.random.uniform(10, 35, num_samples)
weather_conditions = np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], num_samples)
past_traffic_volume = np.random.randint(50, 500, num_samples)  # Past traffic count

# Convert to DataFrame
data = pd.DataFrame({
    'TimeOfDay': time_of_day,
    'Temperature': temperature,
    'Weather': weather_conditions,
    'PastTrafficVolume': past_traffic_volume
})

# Target Variable: Future Traffic Congestion (0 to 100 scale)
data['FutureTrafficCongestion'] = (
    0.5 * data['PastTrafficVolume'] + np.random.uniform(0, 50, num_samples)
)

# Encode Categorical Data (One-Hot Encoding)
encoder = OneHotEncoder(sparse_output=False, drop='first')
weather_encoded = encoder.fit_transform(data[['Weather']])
weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['Weather']))

# Merge Encoded Data
data = pd.concat([data, weather_df], axis=1).drop(columns=['Weather'])

# Train-Test Split
X = data.drop(columns=['FutureTrafficCongestion'])
y = data['FutureTrafficCongestion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Visualization: Actual vs. Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Traffic Congestion')
plt.ylabel('Predicted Traffic Congestion')
plt.title('Actual vs. Predicted Traffic Congestion')
plt.grid(True)
plt.show()
