import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib # Used for loading model and scaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import timedelta
import os

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
DATA_DIR = 'sensor_data/' # Directory containing sensor data
RESULTS_DIR = 'results/'   # Directory containing trained objects and results
ENV_CSV_FILE = 'test_data.csv'          # Generated Feb env data
INCIDENT_REPORTS_FILE = 'incident_reports.csv' # Generated Jan incidents

# Paths to load/save objects (These should point to objects created by train.py)
BASELINE_PATH = os.path.join(RESULTS_DIR, 'baseline_stats.csv')
SCALER_PATH = os.path.join(RESULTS_DIR, 'scaler.joblib')
MODEL_PATH = os.path.join(RESULTS_DIR, 'model.joblib')

# Ensure results directory exists (for saving new results)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Analysis Parameters (from original Analysis.py) ---
ROLLING_WINDOW_SHORT = 30 # Should match training if used in features
ROLLING_WINDOW_LONG = 30  # Should match training if used in features
SLOPE_THRESHOLD = 0.5 # Example threshold, might need tuning
CONSECUTIVE_PREDICTIONS_THRESHOLD = 3

# --- Alert System Parameters ---
IMMEDIATE_ALERT_THRESHOLDS = {
    'temperature': {'max': 20.0, 'min': 13.0}, # Tightened for generated data range
    'humidity': {'max': 63.5, 'min': 61.5},    # Tightened for generated data range
    'gas': {'max': 2500.0, 'min': None},       # Raised significantly for generated data range
    # 'particulates': {'max': 100.0, 'min': None} # Example
}

# --- Output Control ---
VERBOSE = False # Set to True for more detailed print statements (like column lists)
MAX_TABLE_ROWS_TO_PRINT = 5

# --- Data Loading and Preprocessing (for new monitoring data) ---
def load_monitoring_data(file_path):
    """Loads and preprocesses new environmental data for monitoring."""
    print(f"-> Loading Monitoring Data from: '{file_path}'")
    try:
        df = pd.read_csv(
            file_path,
            sep=';',
            decimal=',',
            encoding='utf-8-sig',
            skipinitialspace=True
        )
        if VERBOSE: print(f"  Columns read: {df.columns.tolist()}")
        column_mapping = {
            'TIME': 'timestamp', 'CO2': 'gas',
            'TEMPERATURE': 'temperature', 'HUMIDITY': 'humidity'
        }
        if 'TIME' not in df.columns:
            print("Error: Expected 'TIME' column not found.")
            return None
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        if VERBOSE: print(f"  Columns after rename: {df.columns.tolist()}")
        if 'timestamp' not in df.columns:
            print("Error: Failed to rename timestamp column.")
            return None

        # --- Timezone Handling ---
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            if df.empty:
                print("Error: No valid timestamps after conversion.")
                return None
            if df['timestamp'].dt.tz is None:
                 print("  Warning: Timestamps read without timezone. Assuming UTC and localizing.")
                 # Be careful with ambiguous times if data spans DST changes, 'infer' might be needed
                 df['timestamp'] = df['timestamp'].dt.tz_localize('UTC', ambiguous='raise')
            else:
                 if VERBOSE: print("  Converting environmental timestamps to UTC.")
                 df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        except Exception as e:
            print(f"Error processing environmental timestamps: {e}")
            return None
        # --- End Timezone Handling ---

        df = df.set_index('timestamp')
        df = df.sort_index()

        required_cols = ['temperature', 'humidity', 'gas']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Required column '{col}' missing.")
            elif not pd.api.types.is_numeric_dtype(df[col]):
                 # Added robust conversion just in case
                 if isinstance(df[col].iloc[0], str):
                     df[col] = df[col].str.replace(',', '.', regex=False)
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 if df[col].isnull().any():
                     print(f"  Warning: NaNs introduced in '{col}' during numeric conversion. Filling with ffill/bfill/0.")
                     df[col] = df[col].ffill().bfill().fillna(0) # More robust fill

        print(f"  Monitoring data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df

    except FileNotFoundError:
        print(f"Error: Monitoring data file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error loading monitoring data: {str(e)}")
        return None

# --- Feature Engineering (using loaded baseline) ---
def create_monitoring_features(df, baseline_stats):
    """Creates engineered features using pre-calculated baseline stats."""
    if df is None or df.empty or baseline_stats is None or baseline_stats.empty:
        print("Error: Data or baseline stats missing for feature creation.")
        return None
    print("-> Engineering Features for Monitoring Data...")

    # Sensor columns are in the baseline_stats COLUMNS now
    sensor_cols = [col for col in df.columns if col in baseline_stats.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not sensor_cols:
        print("Error: No common numeric sensor columns found between data and baseline stats columns.")
        if VERBOSE:
            print(f"  Data columns: {df.columns.tolist()}")
            print(f"  Baseline columns: {baseline_stats.columns.tolist()}")
        return None
    if VERBOSE: print(f"  Creating features based on baseline for columns: {sensor_cols}")

    features = df[sensor_cols].copy()
    for col in sensor_cols:
        if col not in baseline_stats.columns:
            print(f"  Warning: Baseline stats missing for column '{col}'. Skipping feature generation for it.")
            continue
        # Access stats using stats as index and sensor name as column key
        mean_val = baseline_stats.loc['mean', col]
        std_dev = baseline_stats.loc['std', col]

        features[f'{col}_baseline_dev'] = features[col] - mean_val
        features[f'{col}_zscore'] = features[f'{col}_baseline_dev'] / std_dev if std_dev > 0 else 0
        features[f'{col}_mavg_short'] = features[col].rolling(window=ROLLING_WINDOW_SHORT, min_periods=1).mean()
        features[f'{col}_mavg_long'] = features[col].rolling(window=ROLLING_WINDOW_LONG, min_periods=1).mean()
        features[f'{col}_roc'] = features[col].diff().fillna(0)
        temp_col = features[col].replace(0, np.nan)
        features[f'{col}_roc_pct'] = temp_col.pct_change().fillna(0) * 100 # Percentage
        features[f'{col}_rstd_short'] = features[col].rolling(window=ROLLING_WINDOW_SHORT, min_periods=1).std().fillna(0)
        features[f'{col}_trend_diff'] = features[f'{col}_mavg_short'] - features[f'{col}_mavg_long']
        features[f'{col}_ewma'] = features[col].ewm(span=ROLLING_WINDOW_SHORT, adjust=False).mean()

    # Handle potential inf/-inf from calculations like pct_change
    features = features.replace([np.inf, -np.inf], np.nan)

    # Use ffill first, then bfill for any remaining at the start
    features = features.ffill().bfill()
    if features.isnull().values.any():
        print("  Warning: NaNs remained after ffill/bfill. Filling with 0.")
        features = features.fillna(0)
    if VERBOSE: print(f"  Created features: {list(features.columns)}")
    print("  Feature engineering complete.")
    return features

# --- Immediate Hazard Check ---
def check_immediate_hazards(df):
    """Checks for immediate safety hazards based on thresholds."""
    if df is None or df.empty:
        print("Error: DataFrame empty, cannot check immediate hazards.")
        return None
    print("\n--- Checking for Immediate Hazards (Threshold Alerts) ---")
    hazard_records = []
    checked_sensors = []
    for col, thresholds in IMMEDIATE_ALERT_THRESHOLDS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            checked_sensors.append(col)
            if thresholds['max'] is not None:
                violations = df.loc[df[col].notna() & (df[col] > thresholds['max'])]
                for idx, row in violations.iterrows():
                    hazard_records.append({'timestamp': idx, 'sensor': col, 'value': row[col], 'threshold': f"> {thresholds['max']}", 'type': 'Maximum Exceeded'})
            if thresholds['min'] is not None:
                violations = df.loc[df[col].notna() & (df[col] < thresholds['min'])]
                for idx, row in violations.iterrows():
                     hazard_records.append({'timestamp': idx, 'sensor': col, 'value': row[col], 'threshold': f"< {thresholds['min']}", 'type': 'Minimum Exceeded'})

    print(f"Checked sensors: {', '.join(checked_sensors)}")
    if hazard_records:
        hazards_df = pd.DataFrame(hazard_records)
        hazards_df = hazards_df.sort_values(by='timestamp')
        print(f"!!! {len(hazards_df)} IMMEDIATE HAZARDS DETECTED !!!")
        print("Sample violations:")
        print(hazards_df.head(MAX_TABLE_ROWS_TO_PRINT).to_string(index=False))
        # Save all violations
        output_path = os.path.join(RESULTS_DIR, 'monitoring_immediate_hazards.csv')
        hazards_df.to_csv(output_path, index=False)
        print(f"Full list saved to: {output_path}")
        return hazards_df
    else:
        print("No immediate hazards detected.")
        return None

# --- Trend Hazard Detection ---
def detect_trend_hazards(df, features, model, scaler):
    """Detects trend-based hazards using the loaded model and scaler."""
    if not all([df is not None, features is not None, model is not None, scaler is not None]):
        print("Error: Missing data, features, model, or scaler for trend hazard detection.")
        return None

    print("\n--- Checking for Trend-Based Hazards (ML Model Alerts) ---")

    # Get feature names EXPECTED by the loaded scaler/model
    try:
        if hasattr(scaler, 'feature_names_in_'):
             expected_features = scaler.feature_names_in_
        elif hasattr(model, 'feature_names_in_'): # Fallback to model if scaler lacks names
             expected_features = model.feature_names_in_
        else:
             # Attempt to infer from number of features if names aren't stored
             if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == len(features.columns) - len([c for c in ['temperature', 'humidity', 'gas'] if c in features.columns]):
                  print("Warning: Scaler/Model missing feature names. Inferring based on column count. Order must be consistent!")
                  base_sensor_cols = [c for c in ['temperature', 'humidity', 'gas'] if c in features.columns]
                  expected_features = [col for col in features.columns if col not in base_sensor_cols]
             else:
                 raise ValueError("Cannot determine expected feature names from scaler or model.")
    except Exception as e:
        print(f"Error getting expected features: {e}. Cannot proceed with prediction.")
        return None

    if VERBOSE: print(f"  Model expects features: {list(expected_features)}")

    # Check if the current features DataFrame has all the columns the model expects
    missing_model_features = [col for col in expected_features if col not in features.columns]
    if missing_model_features:
        print(f"Error: Features required by model are missing from current data: {missing_model_features}")
        return None

    # Ensure order matches if using feature names and select ONLY expected features
    X = features[expected_features]

    # Check for NaNs/Infs before scaling/prediction
    if X.isnull().values.any() or np.isinf(X.values).any():
         print("Warning: NaNs or Infs detected in features before prediction. Attempting to fill.")
         X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
         if X.isnull().values.any():
              print("Error: Could not fill NaNs/Infs in features. Prediction aborted.")
              return None

    try:
        X_scaled = scaler.transform(X)
        all_predictions = model.predict(X_scaled)
        prediction_proba = model.predict_proba(X_scaled)[:, 1]
    except Exception as pred_err:
        print(f"Error during prediction: {pred_err}")
        return None

    results = pd.DataFrame(index=features.index)
    results['risk_prediction'] = all_predictions
    results['risk_probability'] = prediction_proba

    consecutive_count = 0
    trend_alerts = []
    in_alert_state = False
    alert_start_index = -1

    for i in range(len(results)):
        is_risk_predicted = (results['risk_prediction'].iloc[i] == 1)

        if is_risk_predicted:
            if not in_alert_state: # Start of a potential alert sequence
                 alert_start_index = i
            consecutive_count += 1
        else:
            # If we were in an alert state, reset
            if in_alert_state:
                 in_alert_state = False
            consecutive_count = 0
            alert_start_index = -1

        # Trigger alert if threshold met AND we aren't already in this specific alert sequence
        if consecutive_count >= CONSECUTIVE_PREDICTIONS_THRESHOLD and not in_alert_state:
            in_alert_state = True # Mark that we've triggered for this sequence
            alert_time = results.index[i] # Time of the Nth consecutive prediction
            start_alert_time = results.index[alert_start_index] # Time the sequence started

            alert_data = {
                'alert_trigger_time': alert_time,
                'sequence_start_time': start_alert_time,
                'trigger_count': CONSECUTIVE_PREDICTIONS_THRESHOLD,
                'risk_prob_at_trigger': results['risk_probability'].iloc[i]
            }
            # Add sensor values at the time the alert was triggered
            for col in ['temperature', 'humidity', 'gas']:
                if col in df.columns:
                    try:
                        alert_data[col] = df.loc[alert_time, col]
                    except KeyError:
                        alert_data[col] = np.nan
            trend_alerts.append(alert_data)

    if trend_alerts:
        alerts_df = pd.DataFrame(trend_alerts)
        print(f"!!! {len(alerts_df)} TREND-BASED HAZARD SEQUENCES DETECTED !!!")
        print(f"(Based on >= {CONSECUTIVE_PREDICTIONS_THRESHOLD} consecutive high-risk predictions)")
        print("Sample alert sequences:")
        print(alerts_df.head(MAX_TABLE_ROWS_TO_PRINT).to_string(index=False))
        output_path = os.path.join(RESULTS_DIR, 'monitoring_trend_alerts.csv')
        alerts_df.to_csv(output_path, index=False)
        print(f"Full list saved to: {output_path}")
        return alerts_df
    else:
        print("No trend-based hazards detected by the model.")
        return None

# --- Historical Trend Analysis ---
def analyze_historical_trends(df):
    """Analyzes long-term trends in environmental data."""
    if df is None or df.empty:
        print("Error: DataFrame empty, cannot analyze historical trends.")
        return None
    print("\n--- Analyzing Historical Trends ---")

    # Check if data spans enough time for daily resampling
    time_span = df.index.max() - df.index.min()
    if time_span < timedelta(days=1.5): # Require at least 1.5 days for meaningful daily trend
        print(f"Info: Monitoring data spans only {time_span}. Insufficient range for daily trend analysis. Skipping.")
        return None

    trends = {}
    sensor_cols = [col for col in df.select_dtypes(include=np.number).columns if col in ['temperature', 'humidity', 'gas']]
    if not sensor_cols:
        print("No numeric sensor columns found for trend analysis.")
        return None

    # Create a timezone-naive version for resampling if needed (though UTC index is fine)
    df_resample = df.copy()
    # if df_resample.index.tz is not None:
    #     df_resample.index = df_resample.index.tz_localize(None)

    print("Calculating daily average trends...")
    for col in sensor_cols:
        try:
            # Resample to daily mean, keep only days with data
            daily_mean = df_resample[col].resample('D').mean().dropna()

            if len(daily_mean) > 1: # Need at least two points for a trend line
                x = np.arange(len(daily_mean))
                y = daily_mean.values
                slope, intercept = np.polyfit(x, y, 1)
                correlation_matrix = np.corrcoef(x, y)
                # Handle potential NaNs if correlation fails (e.g., constant data)
                r_squared = correlation_matrix[0, 1]**2 if correlation_matrix.shape == (2, 2) and not np.isnan(correlation_matrix).any() else 0.0
                trends[col] = {'slope_per_day': slope, 'intercept': intercept, 'r_squared': r_squared, 'data_points': len(daily_mean)}

                trend_direction = 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable') # Added small tolerance
                significance = 'notable' if abs(slope) > (SLOPE_THRESHOLD * 0.1) and r_squared > 0.1 else 'minimal' # Example significance criteria
                print(f"  {col.capitalize()}: Trend is {trend_direction} ({slope:+.4f} units/day, R2={r_squared:.3f}, based on {len(daily_mean)} daily points) - {significance}")
            else:
                print(f"  {col.capitalize()}: Not enough daily data points ({len(daily_mean)}) for trend analysis.")
        except Exception as e:
            print(f"  Could not analyze trend for {col}: {e}")

    if trends:
        trends_df = pd.DataFrame(trends).T
        output_path = os.path.join(RESULTS_DIR, 'monitoring_trend_analysis.csv')
        trends_df.to_csv(output_path)
        print(f"Trend analysis summary saved to: {output_path}")
    else:
        print("No significant trends calculated.")
    return trends

# --- Incident Correlation ---
def correlate_with_incidents(df):
    """Correlates environmental conditions with incidents, handling timezones."""
    if df is None or df.empty:
        print("Error: DataFrame empty, cannot correlate with incidents.")
        return None
    print("\n--- Correlating Environmental Data with Incidents ---")
    incident_file_path = os.path.join(DATA_DIR, INCIDENT_REPORTS_FILE)
    if not os.path.exists(incident_file_path):
        print(f"Incident report file not found: {incident_file_path}. Skipping correlation.")
        return None

    try:
        incidents = pd.read_csv(incident_file_path)
        if incidents.empty or 'timestamp' not in incidents.columns or 'incident_id' not in incidents.columns:
            print("Incident file empty or missing required columns ('timestamp', 'incident_id'). Skipping correlation.")
            return None

        # --- Timezone Handling ---
        # Environmental data (df) should already be UTC indexed from load_monitoring_data
        if df.index.tz is None:
             print("Error: Environmental data index missing timezone info for correlation.")
             return None # Should not happen

        # Load and convert incident timestamps to UTC
        try:
            incidents['timestamp'] = pd.to_datetime(incidents['timestamp'], errors='coerce')
            incidents = incidents.dropna(subset=['timestamp'])
            if incidents.empty:
                 print("No valid incident timestamps found after conversion. Skipping correlation.")
                 return None
            if incidents['timestamp'].dt.tz is None:
                 print("  Warning: Incident timestamps read without timezone. Assuming UTC and localizing.")
                 incidents['timestamp'] = incidents['timestamp'].dt.tz_localize('UTC', ambiguous='raise')
            else:
                 incidents['timestamp'] = incidents['timestamp'].dt.tz_convert('UTC')
        except Exception as e:
            print(f"Error processing incident timestamps: {e}. Skipping correlation.")
            return None
        # --- End Timezone Handling ---

        min_env_time = df.index.min()
        max_env_time = df.index.max()
        min_inc_time = incidents['timestamp'].min()
        max_inc_time = incidents['timestamp'].max()

        print(f"  Monitoring Data Time Range: {min_env_time} to {max_env_time}")
        print(f"  Incident Data Time Range:   {min_inc_time} to {max_inc_time}")
        print(f"  Loaded {len(incidents)} valid incident reports.")

        hours_before = 1
        incident_conditions = []
        df = df.sort_index() # Ensure sorted

        processed_incident_count = 0
        correlated_incident_count = 0

        for idx, incident in incidents.iterrows():
            incident_time = incident['timestamp']
            window_start = incident_time - timedelta(hours=hours_before)
            window_end = incident_time

            # Check if the incident window could possibly overlap with the env data range
            # (Optimization: skip incidents clearly outside the env data range)
            if incident_time < min_env_time or window_start > max_env_time:
                if VERBOSE: print(f"  Skipping incident {incident['incident_id']} ({incident_time}): outside monitoring data time range.")
                continue

            processed_incident_count +=1
            try:
                # Select data points strictly *before* the incident time
                pre_incident_data = df.loc[window_start:(window_end - timedelta(seconds=1))] # Ensure we don't include the exact incident time
            except Exception as e:
                 # This error is less likely now with the range check above, but keep for safety
                 print(f"Warning: Error slicing data for incident {incident['incident_id']} ({incident_time}): {e}")
                 continue

            if not pre_incident_data.empty:
                numeric_cols = pre_incident_data.select_dtypes(include=np.number).columns
                if not numeric_cols.empty:
                    correlated_incident_count += 1
                    incident_stats = pre_incident_data[numeric_cols].agg(['mean', 'std', 'min', 'max'])
                    sensor_cols_present = [s for s in ['temperature', 'humidity', 'gas'] if s in numeric_cols]
                    for sensor in sensor_cols_present:
                            incident_conditions.append({
                                'incident_id': incident['incident_id'],
                                'incident_type': incident.get('type', 'unknown'),
                                'sensor': sensor,
                                'mean_pre_incident': incident_stats.loc['mean', sensor],
                                'std_pre_incident': incident_stats.loc['std', sensor],
                                'min_pre_incident': incident_stats.loc['min', sensor],
                                'max_pre_incident': incident_stats.loc['max', sensor]
                            })
                # else: # This case is less informative now
                #     if VERBOSE: print(f"  Warning: No numeric data found for incident {incident['incident_id']} in pre-incident window.")

        print(f"Processed {processed_incident_count} incidents potentially overlapping with monitoring data.")
        if incident_conditions:
            conditions_df = pd.DataFrame(incident_conditions)
            print(f"Found environmental data for {correlated_incident_count} incidents in the hour prior.")
            print("\nAverage conditions before incidents (showing mean only):")
            try:
                # Simplified pivot table for readability
                pivot_table = conditions_df.pivot_table(index='incident_type', columns='sensor', values='mean_pre_incident', aggfunc='mean', dropna=False)
                print(pivot_table.round(2).to_string())
                output_path = os.path.join(RESULTS_DIR, 'monitoring_incident_conditions.csv')
                conditions_df.to_csv(output_path, index=False)
                print(f"Detailed pre-incident conditions saved to: {output_path}")
                return conditions_df
            except Exception as pivot_error:
                print(f"Could not create pivot table: {pivot_error}")
                output_path = os.path.join(RESULTS_DIR, 'monitoring_incident_conditions_raw.csv')
                conditions_df.to_csv(output_path, index=False)
                print(f"Raw pre-incident conditions saved to: {output_path}")
                return conditions_df
        else:
            print("No incident data could be correlated with environmental readings within the specified pre-incident window.")
            print("This is expected if the monitoring data and incident report time ranges do not overlap.")
            return None

    except pd.errors.EmptyDataError:
        print(f"Error: Incident report file '{incident_file_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error correlating incidents: {str(e)}")
        return None

# --- Visualization ---
def visualize_data(df, features, model, scaler, baseline_stats):
    """Creates visualizations of monitoring data and predictions."""
    if df is None or df.empty:
        if VERBOSE: print("Visualize: DataFrame empty, cannot create visualizations.")
        return
    print("\n--- Creating Visualizations ---")
    plt.style.use('seaborn-v0_8-darkgrid')
    numeric_sensor_cols = df.select_dtypes(include=np.number).columns.tolist()
    sensor_cols_to_plot = [col for col in numeric_sensor_cols if col in ['temperature', 'humidity', 'gas']]
    if not sensor_cols_to_plot:
        print("Visualize: No sensor columns available to plot.")
        return

    # Add Risk Probability Plot if model exists
    has_risk_plot = (model is not None and features is not None and scaler is not None)
    num_plots = len(sensor_cols_to_plot) + (1 if has_risk_plot else 0)
    if num_plots == 0: return # Nothing to plot

    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes] # Ensure axes is always iterable
    fig.suptitle('Environmental Monitoring and Hazard Detection', fontsize=16)

    # Plot sensor data and overlays
    for i, col in enumerate(sensor_cols_to_plot):
        ax = axes[i]
        try:
            ax.plot(df.index, df[col], label=f'{col.capitalize()}', alpha=0.7)
        except Exception as plot_err:
            print(f"  Visualize: ERROR plotting raw data for {col}: {plot_err}")
            ax.text(0.5, 0.5, f'Error plotting {col}', transform=ax.transAxes, color='red', ha='center', va='center')
            continue
        try:
            # Plot Moving Average
            if features is not None and f'{col}_mavg_short' in features.columns:
                ax.plot(features.index, features[f'{col}_mavg_short'], label=f'{ROLLING_WINDOW_SHORT}pt Avg', linestyle='--', alpha=0.6)
            # Plot Baseline Mean
            if baseline_stats is not None and col in baseline_stats.columns:
                mean_val = baseline_stats.loc['mean', col]
                if pd.notna(mean_val):
                    ax.axhline(y=mean_val, color='gray', linestyle=':', linewidth=1.5, label='Baseline Mean')
            # Plot Immediate Hazard Thresholds
            if col in IMMEDIATE_ALERT_THRESHOLDS:
                max_thresh = IMMEDIATE_ALERT_THRESHOLDS[col].get('max')
                min_thresh = IMMEDIATE_ALERT_THRESHOLDS[col].get('min')
                if max_thresh is not None:
                    ax.axhline(y=max_thresh, color='red', linestyle='-.', linewidth=1.5, label='Max Threshold')
                if min_thresh is not None:
                    ax.axhline(y=min_thresh, color='dodgerblue', linestyle='-.', linewidth=1.5, label='Min Threshold')
        except Exception as overlay_err:
            print(f"  Visualize: Warning - could not plot overlays for {col}: {overlay_err}")

        ax.set_ylabel(col.capitalize())
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Plot risk probability if available
    if has_risk_plot:
        risk_ax = axes[-1]
        try:
            if hasattr(scaler, 'feature_names_in_'): expected_features = scaler.feature_names_in_
            elif hasattr(model, 'feature_names_in_'): expected_features = model.feature_names_in_
            else: raise ValueError("Cannot determine expected features for risk plot.")

            if list(expected_features) and all(c in features.columns for c in expected_features):
                 if hasattr(scaler, 'mean_'): # Check if scaler is fitted
                      X_to_plot = features[expected_features].copy()
                      # Handle potential NaNs/Infs just before plotting prediction
                      X_to_plot = X_to_plot.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
                      X_scaled = scaler.transform(X_to_plot)
                      prediction_proba = model.predict_proba(X_scaled)[:, 1]
                      risk_ax.plot(features.index, prediction_proba, label='Predicted Risk Probability', color='purple', alpha=0.8, linewidth=1.5)
                 else:
                      print("  Visualize: Scaler not fitted, cannot plot risk.")
                      risk_ax.text(0.5, 0.5, 'Risk Plot Failed (Scaler)', transform=risk_ax.transAxes, ha='center', va='center')
            else:
                 print("  Visualize: Features mismatch/missing, cannot plot risk.")
                 risk_ax.text(0.5, 0.5, 'Risk Plot Failed (Features)', transform=risk_ax.transAxes, ha='center', va='center')
        except Exception as risk_plot_err:
            print(f"  Visualize: ERROR plotting risk: {risk_plot_err}")
            risk_ax.text(0.5, 0.5, 'Risk Plot Failed', transform=risk_ax.transAxes, color='red', ha='center', va='center')

        risk_ax.set_ylabel('Risk Probability')
        risk_ax.set_ylim(0, 1)
        risk_ax.legend(loc='upper left', fontsize='small')
        risk_ax.grid(True, linestyle='--', alpha=0.6)

    plt.xlabel("Timestamp")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly
    plot_filename = os.path.join(RESULTS_DIR, 'monitoring_visualization.png')
    try:
        plt.savefig(plot_filename, dpi=150) # Slightly lower DPI for faster saving if needed
        print(f"Monitoring visualization saved to {plot_filename}")
    except Exception as save_err:
        print(f"ERROR saving plot: {save_err}")
    plt.close(fig)

    # Correlation Heatmap
    if len(sensor_cols_to_plot) > 1:
        try:
            corr_matrix = df[sensor_cols_to_plot].corr()
            plt.figure(figsize=(7, 5)) # Smaller heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 10})
            plt.title('Sensor Correlation (Monitoring Data)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            heatmap_filename = os.path.join(RESULTS_DIR, 'monitoring_correlation_heatmap.png')
            plt.savefig(heatmap_filename, dpi=150)
            plt.close()
            print(f"Monitoring correlation heatmap saved to {heatmap_filename}")
        except Exception as corr_err:
            print(f"Could not generate correlation heatmap: {corr_err}")


# --- Summary Function ---
def print_summary(immediate_hazards, trend_hazards, trends, incident_correlations):
    print("\n" + "="*30 + " Monitoring Summary " + "="*30)

    if immediate_hazards is not None and not immediate_hazards.empty:
        print(f"- Immediate Hazards: {len(immediate_hazards)} threshold violations detected.")
    else:
        print("- Immediate Hazards: None detected.")

    if trend_hazards is not None and not trend_hazards.empty:
         print(f"- Trend-Based Hazards: {len(trend_hazards)} alert sequences detected by ML model.")
    else:
         print("- Trend-Based Hazards: No alert sequences detected by ML model.")

    if trends: # Check if the dictionary is not empty
         print("- Historical Trends: Daily trends calculated (see console output and CSV).")
    else:
         print("- Historical Trends: Not calculated (likely due to insufficient data time range).")

    if incident_correlations is not None and not incident_correlations.empty:
        unique_incidents = incident_correlations['incident_id'].nunique()
        print(f"- Incident Correlation: Found pre-incident data for {unique_incidents} incidents.")
    else:
        print("- Incident Correlation: No pre-incident data found matching loaded incidents.")
    print("="*80)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Monitoring Run ---")

    # 1. Load Trained Objects
    baseline_stats = None
    scaler = None
    model = None
    try:
        if not os.path.exists(BASELINE_PATH): raise FileNotFoundError(f"Baseline file not found: {BASELINE_PATH}")
        baseline_stats = pd.read_csv(BASELINE_PATH, index_col=0)
        print(f"-> Loaded baseline stats ({BASELINE_PATH})")

        if not os.path.exists(SCALER_PATH): raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        print(f"-> Loaded scaler ({SCALER_PATH})")

        if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print(f"-> Loaded model ({MODEL_PATH})")

    except FileNotFoundError as e:
        print(f"Error: Trained object not found: {e}")
        print("Please ensure train.py ran successfully and created files in 'results/'.")
        exit()
    except Exception as e:
        print(f"Error loading trained objects: {e}")
        exit()

    # 2. Load New Monitoring Data
    monitor_file_path = os.path.join(DATA_DIR, ENV_CSV_FILE)
    df = load_monitoring_data(monitor_file_path)
    if df is None:
        print("Exiting due to data loading failure.")
        exit()

    # 3. Create Features using loaded baseline
    features = create_monitoring_features(df, baseline_stats)
    if features is None:
        print("Exiting due to feature creation failure.")
        exit()

    # --- Store results from each step ---
    immediate_hazards_result = None
    trend_hazards_result = None
    trends_result = None
    incident_correlations_result = None

    # 4. Perform Monitoring Tasks
    try:
        immediate_hazards_result = check_immediate_hazards(df)
    except Exception as e:
        print(f"\nError during Immediate Hazard Check: {e}")

    try:
        trend_hazards_result = detect_trend_hazards(df, features, model, scaler)
    except Exception as e:
        print(f"\nError during Trend Hazard Detection: {e}")

    try:
        trends_result = analyze_historical_trends(df)
    except Exception as e:
        print(f"\nError during Historical Trend Analysis: {e}")

    try:
        incident_correlations_result = correlate_with_incidents(df)
    except Exception as e:
        print(f"\nError during Incident Correlation: {e}")

    # 5. Create Visualizations
    try:
        visualize_data(df, features, model, scaler, baseline_stats)
    except Exception as e:
         print(f"\nError during Visualization: {e}")

    # 6. Print Final Summary
    print_summary(immediate_hazards_result, trend_hazards_result, trends_result, incident_correlations_result)

    print("\n--- Monitoring Run Finished ---")