import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Sample data generation function for testing
def generate_synthetic_data(num_samples=100):
    np.random.seed(42)
    time_of_day = np.random.randint(0, 24, num_samples)
    temperature = np.random.uniform(10, 35, num_samples)
    weather_conditions = np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], num_samples)
    past_traffic_volume = np.random.randint(50, 500, num_samples)

    data = pd.DataFrame({
        'TimeOfDay': time_of_day,
        'Temperature': temperature,
        'Weather': weather_conditions,
        'PastTrafficVolume': past_traffic_volume
    })

    data['FutureTrafficCongestion'] = (
        0.5 * data['PastTrafficVolume'] + np.random.uniform(0, 50, num_samples)
    )

    return data

# Test Data Integrity
def test_data_integrity():
    data = generate_synthetic_data()
    assert not data.isnull().values.any(), "Data contains missing values"
    assert data.shape[0] > 0, "Dataframe is empty"

# Test Encoding Process
def test_one_hot_encoding():
    data = generate_synthetic_data()
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    weather_encoded = encoder.fit_transform(data[['Weather']])
    
    assert weather_encoded.shape[1] == len(encoder.categories_[0]) - 1, "Encoding dimension mismatch"

# Test Train-Test Split
def test_train_test_split():
    data = generate_synthetic_data()
    X = data.drop(columns=['FutureTrafficCongestion'])
    y = data['FutureTrafficCongestion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) > len(X_test), "Train set should be larger than test set"
    assert len(y_train) > len(y_test), "Train labels should be larger than test labels"

# Test Model Training
def test_model_training():
    data = generate_synthetic_data()
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    weather_encoded = encoder.fit_transform(data[['Weather']])
    weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['Weather']))
    
    # Merge encoded data
    data = pd.concat([data, weather_df], axis=1).drop(columns=['Weather'])

    X = data.drop(columns=['FutureTrafficCongestion'])
    y = data['FutureTrafficCongestion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    assert mse > 0, "MSE should be greater than 0"
    assert 0 <= r2 <= 1, "R2 score should be between 0 and 1"

if __name__ == "__main__":
    pytest.main()
