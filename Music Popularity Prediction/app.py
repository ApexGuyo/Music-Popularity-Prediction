import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset (Replace 'music_data.csv' with your actual dataset)
df = pd.read_csv('music_data.csv')

# Display dataset summary
print(df.head())

# Selecting features and target
features = ['tempo', 'energy', 'danceability', 'loudness',
            'acousticness', 'instrumentalness', 'liveness', 'valence']
target = 'popularity'

X = df[features]
y = df[target]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R2 Score: {r2}')

# Example prediction on new song features
new_song = np.array(
    [[120, 0.8, 0.7, -5, 0.1, 0.0, 0.2, 0.6]])  # Example values
new_song_scaled = scaler.transform(new_song)
predicted_popularity = model.predict(new_song_scaled)
print(f'Predicted Popularity: {predicted_popularity[0]}')
