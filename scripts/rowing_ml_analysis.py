# rowing_ml_analysis.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
password = os.getenv('PASSWORD')

# Database connection setup
engine = create_engine('postgresql://postgres:' + password + '@localhost/rowing-analytics')

def get_data_for_ml(engine):
    """
    Retrieves performance data for machine learning analysis.
    """
    query = """
    SELECT
        r.rower_id,
        r.name AS rower_name,
        r.weight,
        res.time,
        res.margin,
        b.boat_class,
        b.boat_rank,
        p.piece_number,
        e.event_id,
        e.event_date
    FROM
        lineup l
    JOIN
        rower r ON l.rower_id = r.rower_id
    JOIN
        boat b ON l.boat_id = b.boat_id
    JOIN
        result res ON l.piece_id = res.piece_id AND l.boat_id = res.boat_id
    JOIN
        piece p ON res.piece_id = p.piece_id
    JOIN
        event e ON p.event_id = e.event_id
    WHERE
        res.time IS NOT NULL
    ORDER BY
        r.rower_id, p.piece_number;
    """
    df = pd.read_sql_query(query, engine)
    return df

def prepare_data(df):
    """
    Prepares data for machine learning:
    - Feature engineering
    - Handling missing values
    - Encoding categorical variables
    - Splitting into features and labels
    - Scaling features
    """
    # Create a copy to avoid modifying the original dataframe
    df_ml = df.copy()

    # Convert event_date to datetime
    df_ml['event_date'] = pd.to_datetime(df_ml['event_date'])

    # Sort data by rower and event_date
    df_ml.sort_values(['rower_id', 'event_date'], inplace=True)

    # Create lag features (previous times)
    df_ml['prev_time'] = df_ml.groupby('rower_id')['time'].shift(1)
    df_ml['prev_margin'] = df_ml.groupby('rower_id')['margin'].shift(1)

    # Calculate days since last event
    df_ml['days_since_last_event'] = df_ml.groupby('rower_id')['event_date'].diff().dt.days
    df_ml['days_since_last_event'].fillna(0, inplace=True)

    # Drop rows with missing lag features (first entry for each rower)
    df_ml.dropna(subset=['prev_time', 'prev_margin'], inplace=True)

    # Encode categorical variables
    df_ml = pd.get_dummies(df_ml, columns=['boat_class'], drop_first=True)

    # Features and label for performance prediction
    feature_cols = [
        'prev_time',
        'prev_margin',
        'weight',
        'boat_rank',
        'piece_number',
        'days_since_last_event'
    ] + [col for col in df_ml.columns if 'boat_class_' in col]

    X = df_ml[feature_cols].values
    y = df_ml['time'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def build_and_train_model(X_train, y_train, input_dim):
    """
    Builds and trains a neural network model using TensorFlow and Keras.
    """
    # Define the model architecture
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    return model, history

def evaluate_model(model, history, X_test, y_test):
    """
    Evaluates the model on test data and plots training history.
    """
    # Evaluate on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest MAE: {test_mae:.2f} seconds")

    # Plot training & validation loss values
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model Mean Absolute Error')
    plt.ylabel('MAE (seconds)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def make_predictions(model, X_test, y_test):
    """
    Uses the trained model to make predictions and compares them to actual values.
    """
    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Create a DataFrame to compare actual and predicted times
    comparison = pd.DataFrame({
        'Actual Time': y_test,
        'Predicted Time': y_pred
    })

    # Calculate prediction error
    comparison['Error'] = comparison['Actual Time'] - comparison['Predicted Time']

    print("\nComparison of Actual and Predicted Times:")
    print(comparison.head())

    # Plot actual vs. predicted times
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Actual Time', y='Predicted Time', data=comparison)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Time (seconds)')
    plt.ylabel('Predicted Time (seconds)')
    plt.title('Actual vs. Predicted Times')
    plt.show()

def predict_lineup_performance(engine, model, scaler, feature_cols):
    """
    Predicts how well a lineup will perform against another lineup based on rowers.
    """
    # Example: Predicting performance for a new lineup
    # Retrieve the latest data for each rower
    query = """
    SELECT
        r.rower_id,
        r.name AS rower_name,
        r.weight,
        b.boat_rank,
        l.seat_number,
        b.boat_class
    FROM
        lineup l
    JOIN
        rower r ON l.rower_id = r.rower_id
    JOIN
        boat b ON l.boat_id = b.boat_id
    JOIN
        piece p ON l.piece_id = p.piece_id
    WHERE
        p.piece_id = (
            SELECT MAX(piece_id) FROM piece
        )
    ORDER BY
        b.boat_rank, l.seat_number;
    """
    lineup_df = pd.read_sql_query(query, engine)

    # Prepare data for prediction
    # For simplicity, we'll use average previous time and margin for the lineup
    # In practice, you may want to retrieve specific historical data
    avg_prev_time = 0
    avg_prev_margin = 0
    days_since_last_event = 0  # Assuming the event is today

    # Encode boat_class
    boat_class_dummies = pd.get_dummies(lineup_df['boat_class'], drop_first=True)
    for col in [col for col in boat_class_dummies.columns if col not in feature_cols]:
        boat_class_dummies.drop(col, axis=1, inplace=True)

    # Ensure all expected columns are present
    for col in [col for col in feature_cols if 'boat_class_' in col]:
        if col not in boat_class_dummies.columns:
            boat_class_dummies[col] = 0

    # Aggregate data for the lineup
    lineup_features = {
        'prev_time': [avg_prev_time],
        'prev_margin': [avg_prev_margin],
        'weight': [lineup_df['weight'].mean()],
        'boat_rank': [lineup_df['boat_rank'].iloc[0]],
        'piece_number': [1],
        'days_since_last_event': [days_since_last_event]
    }

    lineup_features.update(boat_class_dummies.sum().to_dict())

    lineup_features_df = pd.DataFrame(lineup_features)

    # Reorder columns to match feature_cols
    lineup_features_df = lineup_features_df[feature_cols]

    # Scale features
    lineup_features_scaled = scaler.transform(lineup_features_df.values)

    # Predict performance
    predicted_time = model.predict(lineup_features_scaled).flatten()[0]
    print(f"\nPredicted Time for the Lineup: {predicted_time:.2f} seconds")

def main():
    # Step 1: Retrieve and prepare data
    df = get_data_for_ml(engine)
    if df.empty:
        print("No performance data found.")
        return

    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)

    # Step 2: Build and train the model
    model, history = build_and_train_model(X_train, y_train, input_dim=X_train.shape[1])

    # Step 3: Evaluate the model
    evaluate_model(model, history, X_test, y_test)

    # Step 4: Make predictions and compare
    make_predictions(model, X_test, y_test)

    # Step 5: Predict lineup performance
    predict_lineup_performance(engine, model, scaler, feature_cols)

if __name__ == "__main__":
    main()
