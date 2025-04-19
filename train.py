import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib # Used for saving model and scaler
from datetime import timedelta, timezone # Added timezone
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
DATA_DIR = 'sensor_data/'
RESULTS_DIR = 'results/'
# In train.py
ENV_CSV_FILE = 'training_data.csv'      # Your HISTORICAL Jan data
INC_CSV_FILE = 'incident_reports.csv'  # Generated Jan incidents

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Analysis Parameters ---
BASELINE_PERIODS = 50
ROLLING_WINDOW_SHORT = 30
ROLLING_WINDOW_LONG = 30
STD_DEV_THRESHOLD = 2.0
SLOPE_THRESHOLD = 0.5
RISK_WINDOW_HOURS = 1 # Time window before an incident to label as 'risk'

# --- Data Loading and Preprocessing ---
def load_data(file_path):
    """Loads and preprocesses HISTORICAL environmental data from CSV."""
    try:
        # Attempt to read, assuming the format you provided
        df = pd.read_csv(
            file_path,
            sep=';',
            decimal=',',
            encoding='utf-8-sig',
            skipinitialspace=True
        )
        print(f"Columns read: {df.columns.tolist()}")

        column_mapping = {
            'TIME': 'timestamp', 'CO2': 'gas',
            'TEMPERATURE': 'temperature', 'HUMIDITY': 'humidity'
        }
        if 'TIME' not in df.columns:
            print("Error: Expected 'TIME' column not found.")
            return None

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        print(f"Columns after rename: {df.columns.tolist()}")

        if 'timestamp' not in df.columns:
            print("Error: Failed to rename timestamp column.")
            return None

        # --- Timezone Handling for Environmental Data ---
        try:
            # pd.to_datetime should handle 'YYYY-MM-DDTHH:MM:SS+00:00' automatically
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']) # Remove rows where time conversion failed
            if df.empty:
                print("No valid timestamps found after conversion.")
                return None
            if df['timestamp'].dt.tz is None:
                print("Warning: Timestamps read without timezone. Assuming UTC and localizing.")
                df['timestamp'] = df['timestamp'].dt.tz_localize(timezone.utc, ambiguous='infer')
            else:
                print("Converting environmental timestamps to UTC.")
                df['timestamp'] = df['timestamp'].dt.tz_convert(timezone.utc)
        except Exception as e:
            print(f"Error processing environmental timestamps: {e}")
            return None
        # --- End Timezone Handling ---

        df = df.set_index('timestamp')
        df = df.sort_index() # Ensure chronological order

        required_cols = ['temperature', 'humidity', 'gas']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Required column '{col}' missing after rename.")
                return None
            # Check if data needs conversion even after reading with decimal=','
            if not pd.api.types.is_numeric_dtype(df[col]):
                 # Explicitly replace comma again just in case, then convert
                 if isinstance(df[col].iloc[0], str): # Check if it's actually string
                     df[col] = df[col].str.replace(',', '.', regex=False)
                 df[col] = pd.to_numeric(df[col], errors='coerce')

            if df[col].isnull().any():
                print(f"Warning: NaNs found/introduced in '{col}'. Filling with ffill then 0.")
                # More robust filling for real data
                df[col] = df[col].ffill().fillna(0)

        print(f"Historical environmental data loaded successfully: {len(df)} records (index is UTC)")
        print("Sample of loaded historical data:")
        print(df.head())
        print(df.info()) # Show data types after loading
        return df

    except FileNotFoundError:
        print(f"Error: Historical environmental data file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error loading historical environmental data: {str(e)}")
        return None

# --- Load Incident Data ---
# (Keep this function as is, it should read the new incident file correctly)
def load_incident_data(file_path):
    """Loads and preprocesses incident data from CSV."""
    try:
        incidents = pd.read_csv(
            file_path,
            sep=',',           # Comma separator for incidents
            decimal='.',         # Standard decimal for incidents
            encoding='utf-8-sig',
            skipinitialspace=True
            )
        if incidents.empty or 'timestamp' not in incidents.columns or 'incident_id' not in incidents.columns:
            print("Incident file empty or missing required columns ('timestamp', 'incident_id').")
            return None

        # --- Timezone Handling for Incident Data ---
        try:
            incidents['timestamp'] = pd.to_datetime(incidents['timestamp'], errors='coerce')
            incidents = incidents.dropna(subset=['timestamp']) # Remove rows where time conversion failed
            if incidents.empty:
                 print("No valid incident timestamps after conversion.")
                 return None
            if incidents['timestamp'].dt.tz is None:
                print("Warning: Incident timestamps read without timezone. Assuming UTC and localizing.")
                incidents['timestamp'] = incidents['timestamp'].dt.tz_localize(timezone.utc, ambiguous='infer')
            else:
                print("Converting incident timestamps to UTC.")
                incidents['timestamp'] = incidents['timestamp'].dt.tz_convert(timezone.utc)
        except Exception as e:
            print(f"Error processing incident timestamps: {e}")
            return None
        # --- End Timezone Handling ---

        incidents = incidents.sort_values(by='timestamp')
        print(f"Incident data loaded successfully: {len(incidents)} valid incidents (timestamps converted to UTC)")
        return incidents

    except FileNotFoundError:
        print(f"Error: Incident data file not found at '{file_path}'")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Incident data file '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error loading incident data: {str(e)}")
        return None


# --- Baseline Calculation (Keep as is) ---
def establish_baseline(df):
    """Establishes baseline statistics."""
    if df is None or df.empty:
        print("Error: DataFrame is empty, cannot establish baseline.")
        return None

    baseline_data = df.iloc[:BASELINE_PERIODS] if len(df) >= BASELINE_PERIODS else df
    sensor_cols = [col for col in df.columns if col in ['temperature', 'humidity', 'gas']]

    if not sensor_cols:
        print("Error: No relevant sensor columns found for baseline.")
        return None

    numeric_baseline_data = baseline_data[sensor_cols].select_dtypes(include=np.number)
    if numeric_baseline_data.empty:
        print("Error: No numeric sensor data found in baseline period.")
        return None

    baseline_stats = numeric_baseline_data.agg(['mean', 'std', 'min', 'max'])
    print("\n--- Baseline Statistics (for feature generation) ---")
    print(baseline_stats)
    return baseline_stats

# --- Feature Engineering (Keep as is) ---
def create_features(df, baseline_stats):
    """Creates engineered features from raw sensor data."""
    if df is None or df.empty or baseline_stats is None or baseline_stats.empty:
        print("Error: Data or baseline stats missing for feature creation.")
        return None

    # Use columns present in both df and baseline_stats index for feature generation
    sensor_cols = [col for col in df.columns if col in baseline_stats.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not sensor_cols:
        print("Error: No valid numeric sensor columns found matching baseline stats for feature creation.")
        print(f"DF numeric cols: {df.select_dtypes(include=np.number).columns.tolist()}")
        print(f"Baseline cols: {baseline_stats.columns.tolist()}")
        return None

    print(f"Creating features based on columns: {sensor_cols}")
    features = df[sensor_cols].copy() # Start with just the sensor columns we have stats for
    for col in sensor_cols:
        # Access stats using stats as index and sensor name as column key
        features[f'{col}_baseline_dev'] = features[col] - baseline_stats.loc['mean', col]
        std_dev = baseline_stats.loc['std', col]
        features[f'{col}_zscore'] = features[f'{col}_baseline_dev'] / std_dev if std_dev > 0 else 0
        features[f'{col}_mavg_short'] = features[col].rolling(window=ROLLING_WINDOW_SHORT, min_periods=1).mean()
        features[f'{col}_mavg_long'] = features[col].rolling(window=ROLLING_WINDOW_LONG, min_periods=1).mean()
        features[f'{col}_roc'] = features[col].diff().fillna(0)
        temp_col = features[col].replace(0, np.nan) # Avoid division by zero in pct_change
        features[f'{col}_roc_pct'] = temp_col.pct_change().fillna(0)
        features[f'{col}_rstd_short'] = features[col].rolling(window=ROLLING_WINDOW_SHORT, min_periods=1).std().fillna(0)
        features[f'{col}_trend_diff'] = features[f'{col}_mavg_short'] - features[f'{col}_mavg_long']
        features[f'{col}_ewma'] = features[col].ewm(span=ROLLING_WINDOW_SHORT, adjust=False).mean()

    print("\n--- Feature Engineering Complete ---")
    # Use ffill first, then bfill for any remaining at the start
    features = features.ffill().bfill()
    if features.isnull().values.any():
        print("Warning: NaNs remained after ffill/bfill. Filling with 0.")
        features = features.fillna(0)
    print(f"Created {len(features.columns)} features (including original sensors)")
    return features

# --- Risk Label Generation based on Incidents (Keep as is) ---
def generate_incident_based_labels(environmental_df, incident_df, risk_window_hours):
    """Generates risk labels based on proximity to incidents."""
    if environmental_df is None or environmental_df.empty:
        print("Error: Environmental data missing for label generation.")
        return None
    if incident_df is None or incident_df.empty:
        print("Warning: Incident data missing or empty. Generating all '0' labels.")
        return pd.Series(0, index=environmental_df.index)

    print(f"\n--- Generating Risk Labels based on {risk_window_hours}-hour pre-incident window ---")

    # Ensure both timestamps are timezone-aware (UTC from loading step)
    if environmental_df.index.tz is None or not hasattr(incident_df['timestamp'].iloc[0], 'tzinfo') or incident_df['timestamp'].iloc[0].tzinfo is None :
         print("Error: Timestamps must be timezone-aware (UTC) for comparison.")
         # This shouldn't happen if loading functions worked correctly
         return None

    # Initialize labels as 0 (non-risk)
    labels = pd.Series(0, index=environmental_df.index)
    risk_delta = timedelta(hours=risk_window_hours)

    labeled_count = 0
    incident_timestamps = incident_df['timestamp'].tolist()

    # Efficiently find indices within windows
    env_timestamps = environmental_df.index.to_series() # For faster lookups if needed, though direct index slicing is often fine

    for incident_time in incident_timestamps:
        window_start = incident_time - risk_delta
        window_end = incident_time # Label points *before* the incident

        # Find environmental data points within the window [window_start, window_end)
        mask = (env_timestamps >= window_start) & (env_timestamps < window_end)
        indices_in_window = environmental_df.index[mask]

        if not indices_in_window.empty:
            labels.loc[indices_in_window] = 1
            labeled_count += len(indices_in_window)
            # Optional: print which incident contributed labels
            # print(f"  Labeled {len(indices_in_window)} points for incident at {incident_time}")

    total_labels = len(labels)
    risk_labels_count = labels.sum()
    print(f"Generated labels: {total_labels} total points.")
    print(f"  Labeled {risk_labels_count} points as 'Potential Risk' (1) based on {len(incident_timestamps)} incidents.")
    print(f"  {total_labels - risk_labels_count} points labeled as 'Normal' (0).")

    if risk_labels_count == 0:
        print("Warning: No environmental data points found within the risk window of any incident.")
    elif risk_labels_count < 50: # Arbitrary small number
         print(f"Warning: Very few risk labels ({risk_labels_count}). Model training might be difficult due to imbalance.")

    return labels

# --- Model Training (Keep as is) ---
def train_model(features, labels):
    """Trains the StandardScaler and Logistic Regression model."""
    if features is None or features.empty or labels is None:
        print("Error: Features or labels missing for model training.")
        return None, None
    if len(features) != len(labels):
        print(f"Error: Mismatch between number of feature rows ({len(features)}) and labels ({len(labels)}).")
        return None, None

    print("\n--- Training Risk Prediction Model ---")

    # Identify base sensor columns (which might be in features df)
    base_sensor_cols = ['temperature', 'humidity', 'gas']
    # Select only the *derived* features for training X
    feature_cols_for_training = [col for col in features.columns if col not in base_sensor_cols]

    if not feature_cols_for_training:
        print("Error: No derived features found for training (only base sensor columns remain).")
        return None, None

    print(f"Using {len(feature_cols_for_training)} derived features for training.")
    X = features[feature_cols_for_training]
    y = labels

    # Check for infinite values introduced during feature engineering
    if np.isinf(X).any().any():
        print("Error: Infinite values detected in features before scaling. Replacing with NaN.")
        X = X.replace([np.inf, -np.inf], np.nan)
        # Re-check if NaNs were introduced and fill them
        if X.isnull().any().any():
             print("Warning: NaNs introduced after replacing inf. Filling with 0.")
             X = X.fillna(0) # Simple fill, consider more sophisticated methods if needed

    if len(y.unique()) < 2:
        print(f"Error: Only one class present in labels ({y.unique()}). Cannot train a meaningful classifier.")
        return None, None
    if y.sum() == 0:
         print("Error: All labels are 0. Cannot train model.")
         return None, None


    # Scale features
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        # Store feature names IN THE ORDER THEY WERE USED FOR FITTING
        scaler.feature_names_in_ = feature_cols_for_training
        print("Scaler fitted.")
    except ValueError as ve:
        print(f"Error during scaling: {ve}. Check for non-numeric data or NaNs in features.")
        print(X.info())
        print(X.head())
        # Add check for infinite values again after potential filling
        if np.isinf(X).any().any():
             print("Infinite values still present before scaling.")
        return None, None


    # Train model
    # class_weight='balanced' is important here due to likely imbalance
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000) # Increased max_iter
    try:
        model.fit(X_scaled, y)
         # Store feature names on the model too, for consistency
        model.feature_names_in_ = feature_cols_for_training
        print("Model trained.")
    except Exception as e:
        print(f"Error during model fitting: {e}")
        return None, None

    # Optional: Print feature importance
    if hasattr(model, 'coef_'):
        try:
            # Use the stored feature names from the scaler/model
            importance = pd.DataFrame(model.coef_[0], index=model.feature_names_in_, columns=['Coefficient'])
            importance['Abs_Coefficient'] = importance['Coefficient'].abs()
            importance = importance.sort_values(by='Abs_Coefficient', ascending=False)
            print("\nTop 5 Features by Absolute Coefficient:")
            print(importance[['Coefficient']].head())
        except Exception as e:
            print(f"Could not determine feature importance: {e}")

    return scaler, model


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Model Training ---")

    # 1. Load HISTORICAL Environmental Data
    env_file_path = os.path.join(DATA_DIR, ENV_CSV_FILE)
    df = load_data(env_file_path)
    if df is None:
        print("Exiting due to historical environmental data loading failure.")
        exit()

    # 2. Load ALIGNED Incident Data
    inc_file_path = os.path.join(DATA_DIR, INC_CSV_FILE)
    incidents = load_incident_data(inc_file_path)
    # Allow continuation even if incidents are missing, labels will be all 0 (training will likely fail later, but cleanly)
    if incidents is None:
        print("Proceeding without incident data for labeling. Labels will be all 0.")


    # 3. Establish Baseline (for feature calculation, using HISTORICAL data)
    baseline_stats = establish_baseline(df)
    if baseline_stats is None:
        print("Exiting due to baseline calculation failure.")
        exit()

    # 4. Create Features (using HISTORICAL data and its baseline)
    features = create_features(df, baseline_stats)
    if features is None:
        print("Exiting due to feature creation failure.")
        exit()

    # 5. Generate Labels (using HISTORICAL data and ALIGNED incidents)
    labels = generate_incident_based_labels(df, incidents, RISK_WINDOW_HOURS)
    if labels is None:
        print("Exiting due to label generation failure.")
        exit()

    # Align features and labels (important if data loading/feature creation had issues)
    common_index = features.index.intersection(labels.index)
    if len(common_index) != len(features) or len(common_index) != len(labels):
        print(f"Warning: Aligning features ({len(features)}) and labels ({len(labels)}) to common index ({len(common_index)}).")
        features = features.loc[common_index]
        labels = labels.loc[common_index]
        if features.empty or labels.empty:
             print("Error: No overlapping data between features and labels after alignment.")
             exit()
    else:
         print("Features and Labels successfully aligned.")


    # 6. Train Model
    scaler, model = train_model(features, labels)
    if scaler is None or model is None:
        print("Exiting due to model training failure.")
        exit()

    # 7. Save Objects (Baseline from historical, Scaler/Model trained on historical+aligned incidents)
    try:
        baseline_path = os.path.join(RESULTS_DIR, 'baseline_stats.csv')
        scaler_path = os.path.join(RESULTS_DIR, 'scaler.joblib')
        model_path = os.path.join(RESULTS_DIR, 'model.joblib')

        baseline_stats.to_csv(baseline_path) # Save baseline used for features
        joblib.dump(scaler, scaler_path)
        joblib.dump(model, model_path)

        print(f"\nBaseline stats saved to: {baseline_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Model saved to: {model_path}")

    except Exception as e:
        print(f"\nError saving trained objects: {e}")

    print("\n--- Model Training Finished ---")