from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import numpy as np
import time
import os
import datetime
import threading
import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, db, storage
from sklearn.ensemble import IsolationForest
import pandas as pd

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
stats = {
    'alert_count': 0,
    'alert_change': 0,
    'env_index': 0,
    'env_change': 0
}

sensor_data = {
    'temperature': 0,
    'humidity': 0,
    'air_quality': 0,
    'gas_concentration': 0,
    'last_updated': {
        'temperature': datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'humidity': datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'air_quality': datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'gas_concentration': datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    }
}

env_index_data = {
    'score': 0,
    'components': [],
    'risks': [],
    'recommendations': [],
    'history': [],
    'anomaly_status': 'Normal'
}

alerts = []
next_alert_id = 1

# Firebase Initialization
try:
    firebase_cred_path = "serviceAccountKey.json"
    if os.path.exists(firebase_cred_path):
        cred = credentials.Certificate(firebase_cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://polyufyp-ebfbf-default-rtdb.asia-southeast1.firebasedatabase.app',
            'storageBucket': 'polyufyp-ebfbf.firebasestorage.app'
        })
        db_ref = db.reference()
        bucket = storage.bucket()
        print("Firebase initialized successfully")
        firebase_enabled = True
    else:
        print(f"Firebase credentials file not found at {firebase_cred_path}")
        firebase_enabled = False
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    firebase_enabled = False

# MQTT Configuration
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "fyp_ma5_2025/sensor/"
MQTT_TOPICS = [
    MQTT_TOPIC_PREFIX + "temperature",
    MQTT_TOPIC_PREFIX + "humidity",
    MQTT_TOPIC_PREFIX + "air_quality",
    MQTT_TOPIC_PREFIX + "gas_concentration",
]

mqtt_client = mqtt.Client()
mqtt_connected = False
last_anomaly_alert_time = 0
anomaly_cooldown = 300
consecutive_anomalies = 0
consecutive_anomaly_threshold = 3
iso_forest = IsolationForest(contamination=0.05, random_state=42)
env_data_buffer = []

# MQTT Callbacks
def on_mqtt_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        print(f"Connected to MQTT broker {MQTT_BROKER}")
        mqtt_connected = True
        for topic in MQTT_TOPICS:
            client.subscribe(topic)
            print(f"Subscribed to {topic}")
    else:
        print(f"Failed to connect to MQTT broker, return code {rc}")
        mqtt_connected = False

def on_mqtt_disconnect(client, userdata, rc):
    global mqtt_connected
    print(f"Disconnected from MQTT broker with code {rc}")
    mqtt_connected = False
    threading.Thread(target=connect_mqtt).start()

def on_mqtt_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = msg.payload.decode()
        print(f"Received MQTT message on {topic}: {payload}")
        
        if topic.endswith("temperature"):
            update_sensor("temperature", float(payload))
        elif topic.endswith("humidity"):
            update_sensor("humidity", float(payload))
        elif topic.endswith("air_quality"):
            update_sensor("air_quality", float(payload))
        elif topic.endswith("gas_concentration"):
            update_sensor("gas_concentration", float(payload))
        
        calculate_env_index()
        detect_anomalies()
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_disconnect = on_mqtt_disconnect
mqtt_client.on_message = on_mqtt_message

def connect_mqtt():
    global mqtt_connected
    if not mqtt_connected:
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
            print(f"Connecting to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")

connect_mqtt()

# Firebase helper functions
def save_to_firebase(path, data):
    if not firebase_enabled:
        print("Firebase not enabled, skipping save operation")
        return None
    try:
        ref = db.reference(path)
        ref.set(data)
        return True
    except Exception as e:
        print(f"Error saving to Firebase: {e}")
        return False

def get_from_firebase(path):
    if not firebase_enabled:
        print("Firebase not enabled, skipping get operation")
        return None
    try:
        ref = db.reference(path)
        return ref.get()
    except Exception as e:
        print(f"Error getting data from Firebase: {e}")
        return None

def update_firebase(path, data):
    if not firebase_enabled:
        print("Firebase not enabled, skipping update operation")
        return False
    try:
        ref = db.reference(path)
        ref.update(data)
        return True
    except Exception as e:
        print(f"Error updating Firebase: {e}")
        return False

def push_to_firebase(path, data):
    if not firebase_enabled:
        print("Firebase not enabled, skipping push operation")
        return None
    try:
        ref = db.reference(path)
        new_ref = ref.push(data)
        return new_ref.key
    except Exception as e:
        print(f"Error pushing to Firebase: {e}")
        return None

def delete_from_firebase(path):
    if not firebase_enabled:
        print("Firebase not enabled, skipping delete operation")
        return False
    try:
        ref = db.reference(path)
        ref.delete()
        return True
    except Exception as e:
        print(f"Error deleting from Firebase: {e}")
        return False

# Load data from Firebase on startup
def load_data_from_firebase():
    global alerts, env_index_data, next_alert_id, env_data_buffer
    if not firebase_enabled:
        print("Firebase not enabled, using default data")
        return
    try:
        firebase_alerts = get_from_firebase('alerts')
        if firebase_alerts:
            alerts = []
            alert_ids = []
            for alert_id, alert_data in firebase_alerts.items():
                alerts.append(alert_data)
                if alert_id.startswith('A'):
                    try:
                        alert_num = int(alert_id[1:])
                        alert_ids.append(alert_num)
                    except ValueError:
                        pass
            alerts = sorted(alerts, key=lambda x: x.get('timestamp', 0), reverse=True)
            stats['alert_count'] = len(alerts)
            if alert_ids:
                next_alert_id = max(alert_ids) + 1
            print(f"Loaded {len(alerts)} alerts from Firebase. Next alert ID: A{next_alert_id:03d}")

        firebase_env_index = get_from_firebase('env_index/current')
        if firebase_env_index:
            env_index_data = firebase_env_index
            stats['env_index'] = env_index_data.get('score', 0)
            print("Loaded environmental index data from Firebase")
        else:
            save_to_firebase('env_index/current', env_index_data)
            print("Saved default environmental index to Firebase")

        firebase_env_data = get_from_firebase('environmental_data')
        if firebase_env_data:
            for timestamp, data in firebase_env_data.items():
                env_data_buffer.append([
                    data['temperature'],
                    data['humidity'],
                    data['air_quality']
                ])
            print(f"Loaded {len(env_data_buffer)} historical environmental data points for anomaly detection")

    except Exception as e:
        print(f"Error loading data from Firebase: {e}")

load_data_from_firebase()

def update_sensor(sensor_type, value):
    global sensor_data
    if sensor_type in sensor_data:
        sensor_data[sensor_type] = value
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y/%m/%d %H:%M:%S")
        sensor_data['last_updated'][sensor_type] = timestamp

        socketio.emit('sensor_update', {
            'type': sensor_type,
            'value': value,
            'timestamp': timestamp
        })
        print(f"Updated {sensor_type} sensor: {value}")

def detect_anomalies():
    global env_data_buffer, env_index_data, last_anomaly_alert_time, consecutive_anomalies
    if len(env_data_buffer) < 10:
        return

    data = np.array(env_data_buffer)
    anomaly_scores = iso_forest.fit_predict(data)
    latest_anomaly = anomaly_scores[-1]

    env_index_data['anomaly_status'] = 'Anomaly' if latest_anomaly == -1 else 'Normal'

    if latest_anomaly == -1:
        consecutive_anomalies += 1
    else:
        consecutive_anomalies = 0

    current_time = time.time()
    if (latest_anomaly == -1 and
        consecutive_anomalies >= consecutive_anomaly_threshold and
        env_index_data['score'] < 30 and
        (current_time - last_anomaly_alert_time) > anomaly_cooldown):
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        description = f"Environmental anomaly detected: Temp={sensor_data['temperature']:.2f}°C, Humidity={sensor_data['humidity']:.2f}%, Air Quality={sensor_data['air_quality']:.2f} ppm, Gas Level={sensor_data['gas_concentration']:.0f} ppm"
        add_safety_alert(
            title="Environmental Anomaly Detected",
            location="Environment Monitor",
            alert_type='red',
            description=description
        )
        last_anomaly_alert_time = current_time
        consecutive_anomalies = 0
        print(f"Generated anomaly alert at {timestamp_str}")

    if firebase_enabled:
        save_to_firebase('env_index/current', env_index_data)

    socketio.emit('env_index_update', {
        'score': env_index_data['score'],
        'change': stats['env_change'],
        'components': env_index_data['components'],
        'risks': env_index_data['risks'],
        'anomaly_status': env_index_data['anomaly_status']
    })

def calculate_env_index():
    global env_index_data, stats, sensor_data
    weights = {
        'temperature': 0.25,
        'humidity': 0.25,
        'air_quality': 0.25,
        'gas_concentration': 0.25
    }

    components = []

    temp = sensor_data['temperature']
    if 18 <= temp <= 25:
        temp_score = 100
        temp_status = 'success'
    elif 15 <= temp < 18 or 25 < temp <= 28:
        temp_score = 75
        temp_status = 'warning'
    elif 10 <= temp < 15 or 28 < temp <= 32:
        temp_score = 50
        temp_status = 'warning'
    else:
        temp_score = 25
        temp_status = 'danger'
    components.append({
        'name': 'Temperature',
        'score': temp_score,
        'status': temp_status,
        'weight': weights['temperature'] * 100
    })

    humidity = sensor_data['humidity']
    if 40 <= humidity <= 60:
        humidity_score = 100
        humidity_status = 'success'
    elif 30 <= humidity < 40 or 60 < humidity <= 70:
        humidity_score = 75
        humidity_status = 'warning'
    elif 20 <= humidity < 30 or 70 < humidity <= 80:
        humidity_score = 50
        humidity_status = 'warning'
    else:
        humidity_score = 25
        humidity_status = 'danger'
    components.append({
        'name': 'Humidity',
        'score': humidity_score,
        'status': humidity_status,
        'weight': weights['humidity'] * 100
    })

    air = sensor_data['air_quality']
    if air <= 400:
        air_score = 100
        air_status = 'success'
    elif 400 < air <= 1000:
        air_score = 75
        air_status = 'warning'
    elif 1001 < air <= 2000:
        air_score = 50
        air_status = 'warning'
    else:
        air_score = 25
        air_status = 'danger'
    components.append({
        'name': 'Air Quality',
        ' personally identifiable information score': air_score,
        'status': air_status,
        'weight': weights['air_quality'] * 100
    })

    gas_concentration = sensor_data['gas_concentration']
    if gas_concentration <= 200:
        gas_concentration_score = 100
        gas_concentration_status = 'success'
    elif 201 < gas_concentration <= 500:
        gas_concentration_score = 75
        gas_concentration_status = 'warning'
    elif 501 < gas_concentration <= 800:
        gas_concentration_score = 50
        gas_concentration_status = 'warning'
    else:
        gas_concentration_score = 25
        gas_concentration_status = 'danger'
    components.append({
        'name': 'Gas Level',
        'score': gas_concentration_score,
        'status': gas_concentration_status,
        'weight': weights['gas_concentration'] * 100
    })

    overall_score = int(
        temp_score * weights['temperature'] +
        humidity_score * weights['humidity'] +
        air_score * weights['air_quality'] +
        gas_concentration_score * weights['gas_concentration']
    )

    risks = []
    if temp_score < 50:
        risks.append({
            'name': 'Extreme Temperature',
            'level': 'High' if temp_score <= 25 else 'Medium',
            'status': 'red' if temp_score <= 25 else 'yellow'
        })
    if humidity_score < 50:
        risks.append({
            'name': 'Humidity Issue',
            'level': 'High' if humidity_score <= 25 else 'Medium',
            'status': 'red' if humidity_score <= 25 else 'yellow'
        })
    if air_score < 50:
        risks.append({
            'name': 'Poor Air Quality',
            'level': 'High' if air_score <= 25 else 'Medium',
            'status': 'red' if air_score <= 25 else 'yellow'
        })
    if gas_concentration_score < 50:
        risks.append({
            'name': 'Combustible Gas Detected',
            'level': 'High' if gas_concentration_score <= 25 else 'Medium',
            'status': 'red' if gas_concentration_score <= 25 else 'yellow'
        })

    recommendations = []
    if 'Extreme Temperature' in [r['name'] for r in risks]:
        if temp < 15:
            recommendations.append("Provide additional heating in work areas")
        else:
            recommendations.append("Improve ventilation or add cooling systems")
    if 'Humidity Issue' in [r['name'] for r in risks]:
        if humidity < 30:
            recommendations.append("Use humidifiers to increase moisture in the air")
        else:
            recommendations.append("Improve ventilation and use dehumidifiers")
    if 'High CO₂ Levels' in [r['name'] for r in risks]:
        recommendations.append("Increase fresh air intake and improve ventilation")
        recommendations.append("Check for CO₂ sources and reduce occupancy if necessary")
    if 'Combustible Gas Detected' in [r['name'] for r in risks]:
        recommendations.append("Evacuate the area immediately and check for gas leaks")
        recommendations.append("Ensure all gas detectors are functioning properly")

    old_score = env_index_data.get('score', 0)
    env_index_data['score'] = overall_score
    env_index_data['components'] = components
    env_index_data['risks'] = risks
    env_index_data['recommendations'] = recommendations

    now = datetime.datetime.now()
    history_entry = {
        'date': now.strftime("%m/%d %H:%M"),
        'score': overall_score,
        'timestamp': now.timestamp()
    }
    if 'history' not in env_index_data:
        env_index_data['history'] = []
    env_index_data['history'].append(history_entry)
    if len(env_index_data['history']) > 30:
        env_index_data['history'] = env_index_data['history'][-30:]

    stats['env_index'] = overall_score
    stats['env_change'] = overall_score - old_score

    if firebase_enabled:
        save_to_firebase('env_index/current', env_index_data)
        timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        ref = db.reference(f'environmental_data/{timestamp_str}')
        ref.set({
            'temperature': sensor_data['temperature'],
            'humidity': sensor_data['humidity'],
            'air_quality': sensor_data['air_quality'],
            'safety_index': overall_score
        })

        env_data_buffer.append([
            sensor_data['temperature'],
            sensor_data['humidity'],
            sensor_data['air_quality']
        ])
        if len(env_data_buffer) > 100:
            env_data_buffer.pop(0)

    socketio.emit('env_index_update', {
        'score': overall_score,
        'change': stats['env_change'],
        'components': components,
        'risks': risks,
        'anomaly_status': env_index_data['anomaly_status']
    })
    print(f"Updated environmental index: {overall_score}")

def add_safety_alert(title, location, alert_type='red', description=None):
    global alerts, stats, next_alert_id
    now = datetime.datetime.now()
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

    alerts.insert(0, new_alert)
    if len(alerts) > 100:
        alerts.pop()

    stats['alert_count'] = len(alerts)
    stats['alert_change'] = 1

    if firebase_enabled:
        save_to_firebase(f'alerts/{alert_id}', new_alert)

    unacknowledged_alerts = [a for a in alerts if not a.get('acknowledged', False)][:3]
    socketio.emit('alerts_update', {
        'alerts': unacknowledged_alerts,
        'count': len(unacknowledged_alerts)
    })
    print(f"Added new safety alert: {title} with ID {alert_id}")

@app.route('/')
def index():
    unacknowledged_alerts = [a for a in alerts if not a.get('acknowledged', False)][:3]
    return render_template('index2.html', 
                           stats=stats, 
                           alerts=unacknowledged_alerts,
                           sensor_data=sensor_data,
                           env_index=env_index_data,
                           now=datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

@app.route('/all-alerts')
def all_alerts():
    return render_template('all_alerts2.html', alerts=alerts)

@app.route('/env-details')
def env_details():
    return render_template('env_details.html', env_index=env_index_data)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('get_all_alerts')
def handle_get_all_alerts(data=None):
    return {'alerts': alerts, 'count': len(alerts)}

@socketio.on('get_env_details')
def handle_get_env_details(data=None):
    return env_index_data

@socketio.on('get_sensor_data')
def handle_get_sensor_data(data=None):
    return sensor_data

@socketio.on('acknowledge_alert')
def handle_acknowledge_alert(data):
    alert_id = data.get('alert_id')
    for idx, alert in enumerate(alerts):
        if alert.get('id') == alert_id:
            alerts[idx]['acknowledged'] = True
            if firebase_enabled:
                update_firebase(f'alerts/{alert_id}', {'acknowledged': True})
            unacknowledged_alerts = [a for a in alerts if not a.get('acknowledged', False)][:3]
            socketio.emit('alerts_update', {
                'alerts': unacknowledged_alerts,
                'count': len(alerts)
            })
            return {'success': True}
    return {'success': False, 'error': 'Alert not found'}

@socketio.on('delete_alert')
def handle_delete_alert(data):
    alert_id = data.get('alert_id')
    for idx, alert in enumerate(alerts):
        if alert.get('id') == alert_id:
            del alerts[idx]
            stats['alert_count'] = len(alerts)
            if firebase_enabled:
                delete_from_firebase(f'alerts/{alert_id}')
            unacknowledged_alerts = [a for a in alerts if not a.get('acknowledged', False)][:3]
            socketio.emit('alerts_update', {
                'alerts': unacknowledged_alerts,
                'count': len(alerts)
            })
            return {'success': True}
    return {'success': False, 'error': 'Alert not found'}

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)