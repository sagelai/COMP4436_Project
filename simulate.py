import pandas as pd
import paho.mqtt.client as mqtt
import time
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = 'sensor_data/'
ENV_CSV_FILE = 'test_data.csv'
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "comp4436_gproj/sensor/"
SIMULATION_INTERVAL = 1  # Seconds between updates

# MQTT Topics
MQTT_TOPICS = {
    'temperature': MQTT_TOPIC_PREFIX + "temperature",
    'humidity': MQTT_TOPIC_PREFIX + "humidity",
    'air_quality': MQTT_TOPIC_PREFIX + "air_quality"
}

# MQTT Client
mqtt_client = mqtt.Client()
mqtt_connected = False

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        logger.info(f"Connected to MQTT broker {MQTT_BROKER}")
        mqtt_connected = True
    else:
        logger.error(f"Failed to connect to MQTT broker, return code {rc}")
        mqtt_connected = False

def on_disconnect(client, userdata, rc):
    global mqtt_connected
    logger.info(f"Disconnected from MQTT broker with code {rc}")
    mqtt_connected = False
    connect_mqtt()

def connect_mqtt():
    global mqtt_connected
    if not mqtt_connected:
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
            logger.info(f"Connecting to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")

# Set MQTT callbacks
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

# Load test data
def load_test_data(file_path):
    logger.info(f"Loading test data from: '{file_path}'")
    try:
        df = pd.read_csv(
            file_path,
            sep=';',
            decimal=',',
            encoding='utf-8-sig',
            skipinitialspace=True
        )
        column_mapping = {
            'TIME': 'timestamp',
            'CO2': 'gas',
            'TEMPERATURE': 'temperature',
            'HUMIDITY': 'humidity'
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

        logger.info(f"Test data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    except FileNotFoundError:
        logger.error(f"Test data file not found at '{file_path}'")
        return None
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return None

# Simulate data stream
def simulate_data_stream(test_data):
    global mqtt_connected
    logger.info("Starting data stream simulation")
    current_row_index = 0

    connect_mqtt()
    while True:
        if not mqtt_connected:
            logger.warning("MQTT not connected, retrying...")
            time.sleep(5)
            continue

        # Get current row
        row = test_data.iloc[current_row_index]
        data = {
            'temperature': row['temperature'],
            'humidity': row['humidity'],
            'air_quality': row['gas']  # Publish as air_quality
        }

        # Publish to MQTT
        for sensor, value in data.items():
            topic = MQTT_TOPICS[sensor]
            try:
                mqtt_client.publish(topic, str(value))
                logger.info(f"Published to {topic}: {value}")
            except Exception as e:
                logger.error(f"Error publishing to {topic}: {e}")

        # Move to next row
        current_row_index = (current_row_index + 1) % len(test_data)
        time.sleep(SIMULATION_INTERVAL)

if __name__ == '__main__':
    test_data = load_test_data(os.path.join(DATA_DIR, ENV_CSV_FILE))
    if test_data is None:
        logger.error("Failed to load test_data.csv, exiting")
    else:
        simulate_data_stream(test_data)