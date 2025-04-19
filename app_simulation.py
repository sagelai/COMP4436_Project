from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
from datetime import timedelta, datetime
import os
import time
import threading
import paho.mqtt.client as mqtt
import logging

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
DATA_DIR = 'sensor_data/'
RESULTS_DIR = 'results/'
ENV_CSV_FILE = 'test_data.csv'
INCIDENT_REPORTS_FILE = 'incident_reports.csv'
BASELINE_PATH = os.path.join(RESULTS_DIR, 'baseline_stats.csv')
SCALER_PATH = os.path.join(RESULTS_DIR, 'scaler.joblib')
MODEL_PATH = os.path.join(RESULTS_DIR, 'model.joblib')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Analysis Parameters
ROLLING_WINDOW_SHORT = 30
ROLLING_WINDOW_LONG = 30
SLOPE_THRESHOLD = 0.5
SIMULATION_INTERVAL = 5  # Seconds between updates
ROLLING_WINDOW_SIZE = 50  # Number of recent data points for analysis

# Alert System Parameters
IMMEDIATE_ALERT_THRESHOLDS = {
    'temperature': {'max': 30.0, 'min': 13.0},
    'humidity': {'max': 85, 'min': 61.5},
    'gas': {'max': 2000.0, 'min': 400},
}

VERBOSE = False
MAX_TABLE_ROWS_TO_PRINT = 5
MAX_VIZ_POINTS = 1000  # Limit visualization data points to prevent frontend lag

# Global variables
stats = {
    'alert_count': 0,
    'alert_change': 0
}

sensor_data = {
    'temperature': 0,
    'humidity': 0,
    'air_quality': 0,
    'last_updated': {
        'temperature': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'humidity': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'air_quality': datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    }
}

alerts = []
next_alert_id = 1
analysis_results = {
    'immediate_hazards': None,
    'trend_hazards': None,
    'trends': None,
    'incident_correlations': None,
    'visualization_data': None
}

# Simulation state
test_data = None
current_row_index = 0
rolling_data = pd.DataFrame()  # Rolling window for analysis
simulation_running = False

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "comp4436_gproj/sensor/"
MQTT_TOPICS = [
    MQTT_TOPIC_PREFIX + "temperature",
    MQTT_TOPIC_PREFIX + "humidity",
    MQTT_TOPIC_PREFIX + "air_quality",
]

mqtt_client = mqtt.Client()
mqtt_connected = False

# MQTT Callbacks
def on_mqtt_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        logger.info(f"Connected to MQTT broker {MQTT_BROKER}")
        mqtt_connected = True
        for topic in MQTT_TOPICS:
            client.subscribe(topic)
            logger.info(f"Subscribed to {topic}")
    else:
        logger.error(f"Failed to connect to MQTT broker, return code {rc}")
        mqtt_connected = False

def on_mqtt_disconnect(client, userdata, rc):
    global mqtt_connected
    logger.info(f"Disconnected from MQTT broker with code {rc}")
    mqtt_connected = False
    threading.Thread(target=connect_mqtt).start()

def on_mqtt_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = msg.payload.decode()
        logger.debug(f"Received MQTT message on {topic}: {payload}")
        if topic.endswith("temperature"):
            update_sensor("temperature", float(payload))
        elif topic.endswith("humidity"):
            update_sensor("humidity", float(payload))
        elif topic.endswith("air_quality"):
            update_sensor("air_quality", float(payload))
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_disconnect = on_mqtt_disconnect
mqtt_client.on_message = on_mqtt_message

def connect_mqtt():
    global mqtt_connected
    if not mqtt_connected:
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
            logger.info(f"Connecting to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")

connect_mqtt()

# --- Alert Functions ---
def add_safety_alert(title, location, alert_type='red', description=None):
    global alerts, stats, next_alert_id
    now = datetime.now()
    time_str = f"{now.hour}:{now.minute:02d} {'PM' if now.hour >= 12 else 'AM'}"
    alert_id = f"A{next_alert_id:03d}"
    next_alert_id += 1
    if description is None:
        description = f"{title} at {location}"
    new_alert = {
        'id': alert_id,
        'type': alert_type,
        'title': title,
        'location': location,
        'time': time_str,
        'timestamp': now.timestamp(),
        'acknowledged': False,
        'severity': 'High' if alert_type == 'red' else 'Medium' if alert_type == 'yellow' else 'Low',
        'description': description
    }
    alerts.append(new_alert)
    # Sort alerts: High > Medium > Low, then by timestamp descending
    severity_priority = {'High': 1, 'Medium': 2, 'Low': 3}
    alerts.sort(key=lambda a: (severity_priority.get(a['severity'], 4), -a['timestamp']))
    if len(alerts) > 100:
        alerts.pop()
    stats['alert_count'] = len(alerts)
    stats['alert_change'] = 1
    unacknowledged_alerts = [a for a in alerts if not a.get('acknowledged', False)][:3]
    socketio.emit('alerts_update', {
        'alerts': unacknowledged_alerts,
        'count': len(unacknowledged_alerts)
    })
    logger.info(f"Added safety alert: {title} (ID: {alert_id}, Severity: {new_alert['severity']})")

# --- Analysis Functions ---
def load_monitoring_data(file_path):
    logger.info(f"Loading monitoring data from: '{file_path}'")
    try:
        df = pd.read_csv(
            file_path,
            sep=';',
            decimal=',',
            encoding='utf-8-sig',
            skipinitialspace=True
        )
        column_mapping = {
            'TIME': 'timestamp', 'CO2': 'gas',
            'TEMPERATURE': 'temperature', 'HUMIDITY': 'humidity'
        }
        if 'TIME' not in df.columns:
            logger.error("Expected 'TIME' column not found")
            return None
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        if df.empty:
            logger.error("No valid timestamps after conversion")
            return None
        if df['timestamp'].dt.tz is None:
            logger.warning("Timestamps read without timezone. Assuming UTC")
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC', ambiguous='raise')
        else:
            logger.debug("Converting timestamps to UTC")
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        df = df.set_index('timestamp')
        df = df.sort_index()

        required_cols = ['temperature', 'humidity', 'gas']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' missing")
            elif not pd.api.types.is_numeric_dtype(df[col]):
                if isinstance(df[col].iloc[0], str):
                    df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].ffill().bfill().fillna(0)

        logger.info(f"Monitoring data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    except FileNotFoundError:
        logger.error(f"Monitoring data file not found at '{file_path}'")
        return None
    except Exception as e:
        logger.error(f"Error loading monitoring data: {str(e)}")
        return None

def create_monitoring_features(df, baseline_stats):
    if df is None or df.empty or baseline_stats is None or baseline_stats.empty:
        logger.error("Data or baseline stats missing for feature creation")
        return None
    logger.info("Engineering features for monitoring data")
    sensor_cols = [col for col in df.columns if col in baseline_stats.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not sensor_cols:
        logger.error("No common numeric sensor columns found")
        return None
    logger.debug(f"Creating features for columns: {sensor_cols}")
    features = df[sensor_cols].copy()
    for col in sensor_cols:
        if col not in baseline_stats.columns:
            logger.warning(f"Baseline stats missing for column '{col}'")
            continue
        mean_val = baseline_stats.loc['mean', col]
        std_dev = baseline_stats.loc['std', col]
        features[f'{col}_baseline_dev'] = features[col] - mean_val
        features[f'{col}_zscore'] = features[f'{col}_baseline_dev'] / std_dev if std_dev > 0 else 0
        features[f'{col}_mavg_short'] = features[col].rolling(window=ROLLING_WINDOW_SHORT, min_periods=1).mean()
        features[f'{col}_mavg_long'] = features[col].rolling(window=ROLLING_WINDOW_LONG, min_periods=1).mean()
        features[f'{col}_roc'] = features[col].diff().fillna(0)
        temp_col = features[col].replace(0, np.nan)
        features[f'{col}_roc_pct'] = temp_col.pct_change().fillna(0) * 100
        features[f'{col}_rstd_short'] = features[col].rolling(window=ROLLING_WINDOW_SHORT, min_periods=1).std().fillna(0)
        features[f'{col}_trend_diff'] = features[f'{col}_mavg_short'] - features[f'{col}_mavg_long']
        features[f'{col}_ewma'] = features[col].ewm(span=ROLLING_WINDOW_SHORT, adjust=False).mean()
    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    logger.info(f"Feature engineering complete. Created features: {list(features.columns)}")
    return features

def check_immediate_hazards(df):
    if df is None or df.empty:
        logger.error("DataFrame empty for immediate hazard check")
        return None
    logger.info(f"Checking immediate hazards for columns: {df.columns.tolist()}")
    hazard_records = []
    checked_sensors = []
    for col, thresholds in IMMEDIATE_ALERT_THRESHOLDS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            checked_sensors.append(col)
            logger.debug(f"{col} - Max: {df[col].max()}, Min: {df[col].min()}, Thresholds: {thresholds}")
            if thresholds['max'] is not None:
                violations = df.loc[df[col].notna() & (df[col] > thresholds['max'])]
                logger.debug(f"{col} max violations: {len(violations)}")
                for idx, row in violations.iterrows():
                    description = f"Immediate hazard: {col} = {row[col]:.2f} exceeds max threshold {thresholds['max']}"
                    add_safety_alert(
                        title=f"{col.capitalize()} Threshold Exceeded",
                        location="Environment Monitor",
                        alert_type='red',
                        description=description
                    )
                    hazard_records.append({
                        'timestamp': idx, 'sensor': col, 'value': row[col],
                        'threshold': f"> {thresholds['max']}", 'type': 'Maximum Exceeded'
                    })
            if thresholds['min'] is not None:
                violations = df.loc[df[col].notna() & (df[col] < thresholds['min'])]
                logger.debug(f"{col} min violations: {len(violations)}")
                for idx, row in violations.iterrows():
                    description = f"Immediate hazard: {col} = {row[col]:.2f} below min threshold {thresholds['min']}"
                    add_safety_alert(
                        title=f"{col.capitalize()} Threshold Violated",
                        location="Environment Monitor",
                        alert_type='red',
                        description=description
                    )
                    hazard_records.append({
                        'timestamp': idx, 'sensor': col, 'value': row[col],
                        'threshold': f"< {thresholds['min']}", 'type': 'Minimum Exceeded'
                    })
    logger.info(f"Checked sensors: {', '.join(checked_sensors)}")
    if hazard_records:
        hazards_df = pd.DataFrame(hazard_records).sort_values(by='timestamp')
        output_path = os.path.join(RESULTS_DIR, 'monitoring_immediate_hazards.csv')
        hazards_df.to_csv(output_path, index=False)
        logger.info(f"{len(hazards_df)} immediate hazards detected. Saved to {output_path}")
        return hazards_df
    logger.info("No immediate hazards detected")
    return None

def detect_trend_hazards(df, features, model, scaler):
    if not all([df is not None, features is not None, model is not None, scaler is not None]):
        logger.error("Missing data, features, model, or scaler for trend hazard detection")
        return None
    logger.info("Checking for trend-based hazards")
    expected_features = getattr(scaler, 'feature_names_in_', None) or getattr(model, 'feature_names_in_', None)
    if not expected_features:
        logger.error("Cannot determine expected feature names")
        return None
    missing_features = [col for col in expected_features if col not in features.columns]
    if missing_features:
        logger.error(f"Features required by model missing: {missing_features}")
        return None
    X = features[expected_features]
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    try:
        X_scaled = scaler.transform(X)
        all_predictions = model.predict(X_scaled)
        prediction_proba = model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None
    results = pd.DataFrame(index=features.index)
    results['risk_prediction'] = all_predictions
    results['risk_probability'] = prediction_proba

    trend_alerts = []
    for i in range(len(results)):
        if results['risk_prediction'].iloc[i] == 1:
            alert_time = results.index[i]
            description = f"Trend-based hazard detected at {alert_time} with risk probability {results['risk_probability'].iloc[i]:.2f}"
            add_safety_alert(
                title="Trend-Based Hazard Detected",
                location="Environment Monitor",
                alert_type='yellow',
                description=description
            )
            alert_data = {
                'alert_trigger_time': alert_time,
                'risk_prob_at_trigger': results['risk_probability'].iloc[i]
            }
            for col in ['temperature', 'humidity', 'air_quality']:
                if col in df.columns:
                    try:
                        alert_data[col] = df.loc[alert_time, col]
                    except KeyError:
                        alert_data[col] = np.nan
            trend_alerts.append(alert_data)
    
    if trend_alerts:
        alerts_df = pd.DataFrame(trend_alerts)
        output_path = os.path.join(RESULTS_DIR, 'monitoring_trend_alerts.csv')
        alerts_df.to_csv(output_path, index=False)
        logger.info(f"{len(alerts_df)} trend-based hazards detected. Saved to {output_path}")
        return alerts_df
    logger.info("No trend-based hazards detected")
    return None

def analyze_historical_trends(df):
    if df is None or df.empty:
        logger.error("DataFrame empty for trend analysis")
        return None
    logger.info("Analyzing historical trends")
    time_span = df.index.max() - df.index.min()
    if time_span < timedelta(days=1.5):
        logger.info(f"Data spans only {time_span}. Insufficient for trend analysis")
        return None
    trends = {}
    sensor_cols = [col for col in df.select_dtypes(include=np.number).columns if col in ['temperature', 'humidity', 'air_quality']]
    for col in sensor_cols:
        daily_mean = df[col].resample('D').mean().dropna()
        if len(daily_mean) > 1:
            x = np.arange(len(daily_mean))
            y = daily_mean.values
            slope, intercept = np.polyfit(x, y, 1)
            correlation_matrix = np.corrcoef(x, y)
            r_squared = correlation_matrix[0, 1]**2 if correlation_matrix.shape == (2, 2) and not np.isnan(correlation_matrix).any() else 0.0
            trends[col] = {'slope_per_day': slope, 'intercept': intercept, 'r_squared': r_squared, 'data_points': len(daily_mean)}
            logger.info(f"{col.capitalize()}: Trend {slope:+.4f} units/day, R2={r_squared:.3f}")
        else:
            logger.info(f"{col.capitalize()}: Not enough data points")
    if trends:
        trends_df = pd.DataFrame(trends).T
        output_path = os.path.join(RESULTS_DIR, 'monitoring_trend_analysis.csv')
        trends_df.to_csv(output_path)
        logger.info(f"Trends saved to {output_path}")
    return trends

def correlate_with_incidents(df):
    if df is None or df.empty:
        logger.error("DataFrame empty for incident correlation")
        return None
    logger.info("Correlating with incidents")
    incident_file_path = os.path.join(DATA_DIR, INCIDENT_REPORTS_FILE)
    if not os.path.exists(incident_file_path):
        logger.warning(f"Incident file not found: {incident_file_path}")
        return None
    try:
        incidents = pd.read_csv(incident_file_path)
        if incidents.empty or 'timestamp' not in incidents.columns:
            logger.warning("Incident file empty or missing 'timestamp'")
            return None
        incidents['timestamp'] = pd.to_datetime(incidents['timestamp'], errors='coerce')
        incidents = incidents.dropna(subset=['timestamp'])
        if incidents['timestamp'].dt.tz is None:
            logger.warning("Incident timestamps without timezone. Assuming UTC")
            incidents['timestamp'] = incidents['timestamp'].dt.tz_localize('UTC', ambiguous='raise')
        else:
            incidents['timestamp'] = incidents['timestamp'].dt.tz_convert('UTC')
        hours_before = 1
        incident_conditions = []
        for idx, incident in incidents.iterrows():
            incident_time = incident['timestamp']
            window_start = incident_time - timedelta(hours=hours_before)
            window_end = incident_time
            if incident_time < df.index.min() or window_start > df.index.max():
                continue
            pre_incident_data = df.loc[window_start:(window_end - timedelta(seconds=1))]
            if not pre_incident_data.empty:
                numeric_cols = pre_incident_data.select_dtypes(include=np.number).columns
                if numeric_cols.empty:
                    continue
                incident_stats = pre_incident_data[numeric_cols].agg(['mean', 'std', 'min', 'max'])
                sensor_cols_present = [s for s in ['temperature', 'humidity', 'air_quality'] if s in numeric_cols]
                for sensor in sensor_cols_present:
                    incident_conditions.append({
                        'incident_id': incident.get('incident_id', 'unknown'),
                        'incident_type': incident.get('type', 'unknown'),
                        'sensor': sensor,
                        'mean_pre_incident': incident_stats.loc['mean', sensor],
                        'std_pre_incident': incident_stats.loc['std', sensor],
                        'min_pre_incident': incident_stats.loc['min', sensor],
                        'max_pre_incident': incident_stats.loc['max', sensor]
                    })
        if incident_conditions:
            conditions_df = pd.DataFrame(incident_conditions)
            output_path = os.path.join(RESULTS_DIR, 'monitoring_incident_conditions.csv')
            conditions_df.to_csv(output_path, index=False)
            logger.info(f"Pre-incident conditions saved to {output_path}")
            return conditions_df
        logger.info("No incident correlations found")
        return None
    except Exception as e:
        logger.error(f"Error correlating incidents: {str(e)}")
        return None

def prepare_visualization_data(df, features, model, scaler, baseline_stats):
    if df is None or df.empty:
        logger.error("DataFrame empty for visualization data")
        return None
    logger.info("Preparing visualization data")
    sensor_cols = ['temperature', 'humidity', 'air_quality']
    
    # Downsample if too many points
    if len(df) > MAX_VIZ_POINTS:
        logger.debug(f"Downsampling visualization data from {len(df)} to {MAX_VIZ_POINTS} points")
        df = df.iloc[::len(df)//MAX_VIZ_POINTS].iloc[:MAX_VIZ_POINTS]
        if features is not None:
            features = features.loc[df.index]

    # Rename gas to air_quality for visualization
    df_viz = df.rename(columns={'gas': 'air_quality'})
    baseline_stats_viz = baseline_stats.rename(columns={'gas': 'air_quality'})

    viz_data = {
        'timestamps': [t.isoformat() for t in df_viz.index],
        'sensors': {col: df_viz[col].tolist() for col in sensor_cols if col in df_viz.columns},
        'moving_averages': {},
        'baselines': {},
        'thresholds': {},
        'risk_probability': []
    }
    for col in sensor_cols:
        if col in df_viz.columns and features is not None and f'{col}_mavg_short' in features.columns:
            viz_data['moving_averages'][col] = features[f'{col}_mavg_short'].tolist()
        if col in df_viz.columns and baseline_stats_viz is not None and col in baseline_stats_viz.columns:
            mean_val = baseline_stats_viz.loc['mean', col]
            if pd.notna(mean_val):
                viz_data['baselines'][col] = [mean_val] * len(df_viz)
        # Adjust thresholds for visualization
        viz_col = 'air_quality' if col == 'air_quality' else col
        if viz_col in IMMEDIATE_ALERT_THRESHOLDS:
            viz_data['thresholds'][col] = {
                'max': IMMEDIATE_ALERT_THRESHOLDS[viz_col].get('max'),
                'min': IMMEDIATE_ALERT_THRESHOLDS[viz_col].get('min')
            }
    if model is not None and features is not None and scaler is not None:
        expected_features = getattr(scaler, 'feature_names_in_', None) or getattr(model, 'feature_names_in_', None)
        if expected_features and all(c in features.columns for c in expected_features):
            X = features[expected_features].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            try:
                X_scaled = scaler.transform(X)
                viz_data['risk_probability'] = model.predict_proba(X_scaled)[:, 1].tolist()
            except Exception as e:
                logger.error(f"Error generating risk probabilities for visualization: {e}")
    return viz_data

# --- Simulation Functions ---
def simulate_data_stream():
    global test_data, current_row_index, rolling_data, simulation_running
    logger.info("Starting data stream simulation")
    simulation_running = True
    try:
        baseline_stats = pd.read_csv(BASELINE_PATH, index_col=0)
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded trained objects for simulation")
    except Exception as e:
        logger.error(f"Error loading trained objects: {e}")
        simulation_running = False
        return

    while simulation_running and test_data is not None:
        # Get current row
        row = test_data.iloc[current_row_index]
        timestamp = row.name
        data = {
            'temperature': row['temperature'],
            'humidity': row['humidity'],
            'gas': row['gas']
        }

        # Update sensor_data
        now = datetime.now()
        timestamp_str = now.strftime("%Y/%m/%d %H:%M:%S")
        for sensor in ['temperature', 'humidity', 'gas']:
            sensor_data[sensor if sensor != 'gas' else 'air_quality'] = data[sensor]
            sensor_data['last_updated'][sensor if sensor != 'gas' else 'air_quality'] = timestamp_str
            socketio.emit('sensor_update', {
                'type': sensor if sensor != 'gas' else 'air_quality',
                'value': data[sensor],
                'timestamp': timestamp_str
            })
            logger.debug(f"Updated {sensor} sensor: {data[sensor]}")

        # Append to rolling_data
        new_row = pd.DataFrame([data], index=[timestamp])
        global rolling_data
        rolling_data = pd.concat([rolling_data, new_row])
        if len(rolling_data) > ROLLING_WINDOW_SIZE:
            rolling_data = rolling_data.iloc[-ROLLING_WINDOW_SIZE:]
        logger.debug(f"Rolling data: {len(rolling_data)} rows, latest: {rolling_data.iloc[-1].to_dict()}")

        # Perform analysis on rolling_data
        df_analysis = rolling_data.copy()
        df_viz = rolling_data.copy()  # Keep gas for immediate hazards
        baseline_stats_viz = baseline_stats
        temp_thresholds = IMMEDIATE_ALERT_THRESHOLDS

        # Feature generation
        features = create_monitoring_features(df_analysis, baseline_stats)
        if features is None:
            logger.error("Feature creation failed in simulation")
            time.sleep(SIMULATION_INTERVAL)
            current_row_index = (current_row_index + 1) % len(test_data)
            continue

        # Immediate hazards
        try:
            analysis_results['immediate_hazards'] = check_immediate_hazards(df_viz)
        except Exception as e:
            logger.error(f"Error in immediate hazard check: {e}")

        # Trend-based hazards
        try:
            analysis_results['trend_hazards'] = detect_trend_hazards(df_viz, features, model, scaler)
        except Exception as e:
            logger.error(f"Error in trend hazard detection: {e}")

        # Visualization data
        try:
            viz_data = prepare_visualization_data(df_viz, features, model, scaler, baseline_stats_viz)
            if viz_data:
                analysis_results['visualization_data'] = viz_data
                socketio.emit('visualization_update', viz_data)
                logger.debug("Emitted visualization update")
        except Exception as e:
            logger.error(f"Error preparing visualization data: {e}")

        # Move to next row
        current_row_index = (current_row_index + 1) % len(test_data)
        time.sleep(SIMULATION_INTERVAL)

    logger.info("Simulation stopped")

def start_simulation():
    global test_data
    test_data = load_monitoring_data(os.path.join(DATA_DIR, ENV_CSV_FILE))
    if test_data is None:
        logger.error("Failed to load test_data.csv for simulation")
        return
    logger.info(f"Loaded test_data.csv with {len(test_data)} rows")
    threading.Thread(target=simulate_data_stream, daemon=True).start()

# --- Other Functions ---
def update_sensor(sensor_type, value):
    global sensor_data
    if sensor_type in sensor_data:
        sensor_data[sensor_type] = value
        now = datetime.now()
        timestamp = now.strftime("%Y/%m/%d %H:%M:%S")
        sensor_data['last_updated'][sensor_type] = timestamp
        socketio.emit('sensor_update', {
            'type': sensor_type,
            'value': value,
            'timestamp': timestamp
        })
        logger.debug(f"Updated {sensor_type} sensor: {value}")

# Flask Routes
@app.route('/')
def index():
    severity_priority = {'High': 1, 'Medium': 2, 'Low': 3}
    unacknowledged_alerts = sorted(
        [a for a in alerts if not a.get('acknowledged', False)],
        key=lambda a: (severity_priority.get(a['severity'], 4), -a['timestamp'])
    )[:3]
    return render_template('index.html', 
                           stats=stats, 
                           alerts=unacknowledged_alerts,
                           sensor_data=sensor_data,
                           now=datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

@app.route('/all-alerts')
def all_alerts():
    severity_priority = {'High': 1, 'Medium': 2, 'Low': 3}
    sorted_alerts = sorted(
        alerts,
        key=lambda a: (severity_priority.get(a['severity'], 4), -a['timestamp'])
    )
    return render_template('all_alerts.html', alerts=sorted_alerts)

@app.route('/visualization-data')
def get_visualization_data():
    viz_data = analysis_results.get('visualization_data', {})
    if not viz_data:
        logger.warning("No visualization data available")
    return jsonify(viz_data)

# SocketIO Handlers
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('get_all_alerts')
def handle_get_all_alerts(data=None):
    severity_priority = {'High': 1, 'Medium': 2, 'Low': 3}
    sorted_alerts = sorted(
        alerts,
        key=lambda a: (severity_priority.get(a['severity'], 4), -a['timestamp'])
    )
    return {'alerts': sorted_alerts, 'count': len(alerts)}

@socketio.on('get_sensor_data')
def handle_get_sensor_data(data=None):
    return sensor_data

@socketio.on('acknowledge_alert')
def handle_acknowledge_alert(data):
    alert_id = data.get('alert_id')
    severity_priority = {'High': 1, 'Medium': 2, 'Low': 3}
    for idx, alert in enumerate(alerts):
        if alert.get('id') == alert_id:
            alerts[idx]['acknowledged'] = True
            unacknowledged_alerts = sorted(
                [a for a in alerts if not a.get('acknowledged', False)],
                key=lambda a: (severity_priority.get(a['severity'], 4), -a['timestamp'])
            )[:3]
            socketio.emit('alerts_update', {
                'alerts': unacknowledged_alerts,
                'count': len(unacknowledged_alerts)
            })
            logger.info(f"Acknowledged alert: {alert_id}")
            return {'success': True}
    logger.warning(f"Alert not found for acknowledgement: {alert_id}")
    return {'success': False, 'error': 'Alert not found'}

@socketio.on('delete_alert')
def handle_delete_alert(data):
    alert_id = data.get('alert_id')
    severity_priority = {'High': 1, 'Medium': 2, 'Low': 3}
    for idx, alert in enumerate(alerts):
        if alert.get('id') == alert_id:
            del alerts[idx]
            stats['alert_count'] = len(alerts)
            unacknowledged_alerts = sorted(
                [a for a in alerts if not a.get('acknowledged', False)],
                key=lambda a: (severity_priority.get(a['severity'], 4), -a['timestamp'])
            )[:3]
            socketio.emit('alerts_update', {
                'alerts': unacknowledged_alerts,
                'count': len(unacknowledged_alerts)
            })
            logger.info(f"Deleted alert: {alert_id}")
            return {'success': True}
    logger.warning(f"Alert not found for deletion: {alert_id}")
    return {'success': False, 'error': 'Alert not found'}

# Start simulation on startup
if __name__ == '__main__':
    start_simulation()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)