# Construction Site Environmental Monitoring System

This is a web-based application for real-time environmental monitoring and safety alerts on construction sites. It integrates with MQTT for sensor data, Firebase for data storage, and uses machine learning for anomaly detection. The UI displays environmental sensor data (temperature, humidity, CO₂ levels, gas concentration), an environmental safety index, and safety alerts, built with Tailwind CSS, Chart.js, and Socket.io for real-time updates.

## Prerequisites

- **Python 3.8+**: Ensure Python is installed.
- **pip**: Python package manager for installing dependencies.
- **Firebase Account**: Required for database and storage integration.
- **MQTT Broker**: An MQTT broker (e.g., Mosquitto) for sensor communication.
- A modern web browser (e.g., Chrome, Firefox).

## Setup Instructions

Follow these steps to set up and run the application:

### 1. Clone the Repository

Clone the repository or ensure you have the project files (`app.py`, `templates/index.html`, `templates/all_alerts.html`, etc.) in a project directory.

```bash
git clone <repository-url>
cd construction-site-monitoring
```

### 2. Create a Virtual Environment

Create and activate a Python virtual environment to manage dependencies.

```bash
python -m venv venv
```

- **Windows**:

  ```bash
  venv\Scripts\activate
  ```

- **MacOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required Python packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

- `Flask`: Web framework.
- `Flask-SocketIO`, `python-socketio`, `eventlet`: For real-time communication.
- `numpy`, `pandas`, `scikit-learn`: For data processing and anomaly detection.
- `paho-mqtt`: For MQTT communication with sensors.
- `firebase-admin`: For Firebase database and storage.

### 4. Configure Firebase

1. Create a Firebase project at console.firebase.google.com.

2. Generate a service account key:

   - Go to Project Settings &gt; Service Accounts &gt; Generate New Private Key.
   - Download the JSON key file (e.g., `serviceAccountKey.json`).

3. Place the JSON key file in the project directory.

4. Update `app.py` to point to the correct path of the Firebase credentials file, e.g.:

   ```python
   cred = credentials.Certificate("path/to/serviceAccountKey.json")
   firebase_admin.initialize_app(cred, {
       'databaseURL': '<your-firebase-database-url>',
       'storageBucket': '<your-firebase-storage-bucket>'
   })
   ```

5. Ensure Firebase Realtime Database and Storage rules are configured to allow read/write access as needed.

### 5. Configure MQTT Broker

1. Set up an MQTT broker (e.g., Mosquitto) locally or on a server.

2. Update `app.py` with the correct broker address and port, e.g.:

   ```python
   client = mqtt.Client()
   client.connect("localhost", 1883, 60)
   ```

3. Ensure sensors or devices are publishing data to the MQTT topics subscribed to in `app.py`.

### 6. Project Structure

Ensure your project directory has the following structure:

```
construction-site-monitoring/
├── static/
│   └── (custom CSS/JS, if any)
├── templates/
│   ├── index.html
│   └── all_alerts.html
├── app.py
├── serviceAccountKey.json
├── requirements.txt
└── README.md
```

- `app.py`: Main Flask application handling routes, Socket.io, MQTT, and Firebase.
- `templates/`: HTML templates (`index.html`, `all_alerts.html`).
- `static/`: For custom static files (not required since Tailwind CSS, Chart.js, etc., are CDN-based).
- `serviceAccountKey.json`: Firebase service account credentials.

### 7. Run the Application

Start the Flask application with the virtual environment activated.

```bash
python app.py
```

The application typically runs on `http://localhost:5000`. Check `app.py` for the configured port if different.

### 8. Open the Application

Open a web browser and navigate to:

```
http://localhost:5000
```

You should see the dashboard with:

- Environmental monitoring data (temperature, humidity, CO₂, gas concentration).
- Environmental safety index with a gauge and risk details.
- Safety alerts with real-time updates.

Navigate to `/all-alerts` to view and filter all alerts by date, location, or severity.

### 9. Accessing Features

- **Environmental Monitoring**: Real-time sensor data from MQTT.
- **Environmental Safety Index**: Composite safety score with anomaly detection (via IsolationForest).
- **Safety Alerts**: View, acknowledge, or delete alerts stored in Firebase.
- **All Alerts Page**: Filter alerts by status, date, location, or severity.

## Troubleshooting

- **Firebase Errors**: Verify the `serviceAccountKey.json` path and Firebase URLs in `app.py`. Check Firebase console for database/storage rule issues.
- **MQTT Issues**: Ensure the MQTT broker is running and accessible. Verify topic subscriptions in `app.py`.
- **Dependency Conflicts**: Update pip (`pip install --upgrade pip`) and retry installing `requirements.txt`.
- **No Data Displayed**: Confirm MQTT sensors are publishing data and Firebase is receiving updates. Check `app.py` Socket.io events (`sensor_update`, `env_index_update`, `alerts_update`).
- **Port Conflict**: If `port 5000` is in use, modify `app.py` to use another port, e.g., `app.run(port=5001)`.

## Notes

- The frontend uses CDN-hosted libraries (Tailwind CSS, Chart.js, Socket.io, Flatpickr), so no Node.js setup is required.
- Ensure `app.py` handles Flask routes (`/`, `/all-alerts`), Socket.io events, MQTT subscriptions, and Firebase interactions.
- The application supports dark mode via a theme toggle.
- Machine learning (IsolationForest) requires sufficient data for anomaly detection; ensure sensor data is collected in Firebase or processed in `app.py`.

For further assistance, refer to the Flask, Flask-SocketIO, Firebase Admin, or Paho MQTT documentation, or contact the project maintainer.
