## System Components

1. **Data Generation Script (`generate_data.py`)**: Generates synthetic training, test, and incident report data for model training and testing.
2. **Model Training Script (`train.py`)**: Trains a LogisticRegression model to detect trend-based hazards using environmental and incident data.
3. **Flask Application (`app.py`)**: Subscribes to MQTT topics, processes sensor data, generates immediate and trend-based hazard alerts, and provides a web dashboard for visualization.

## Prerequisites

- **Software**:
  - Python 3.7+
  - Internet connection to subscribe to the MQTT broker (`broker.emqx.io`)
  - Web browser to access the Flask dashboard
- **Data Source**:
  - Environmental sensor data published to MQTT topics:
    - `comp4436_gproj/sensor/temperature`
    - `comp4436_gproj/sensor/humidity`
    - `comp4436_gproj/sensor/air_quality`
  - Ensure the MQTT broker is accessible and data is being published (e.g., by an external sensor system).

## Installation

1. Clone or download this project to your computer.

2. Navigate to the project directory:

   ```bash
   cd /path/to/project
   ```

3. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Setup

### 1. Generate Data (`generate_data.py`)

This script creates synthetic environmental and incident data for training and testing the model.

1. Run the data generation script:

   ```bash
   python generate_data.py
   ```

2. Outputs:
   - `sensor_data/test_data.csv`: Synthetic environmental data for testing/monitoring
   - `sensor_data/incident_reports.csv`: Synthetic incident reports for training (aligned with Jan 18–22, 2023)

3. Note: If you have historical environmental data, place it as `sensor_data/training_data.csv`. Otherwise, use synthetic data from this script.

### 2. Train the Model (`train.py`)

This script trains a LogisticRegression model for trend-based hazard detection.

1. Ensure `sensor_data/` contains:
   - `training_data.csv` (historical or synthetic environmental data)
   - `incident_reports.csv` (from `generate_data.py`)

2. Run the training script:

   ```bash
   python train.py
   ```

3. Outputs in `results/`:
   - `baseline_stats.csv`: Baseline statistics for sensor data
   - `scaler.joblib`: Trained StandardScaler
   - `model.joblib`: Trained LogisticRegression model

### 3. Set Up the Flask Application (`app.py`)

1. Ensure `results/` contains:
   - `baseline_stats.csv`
   - `scaler.joblib`
   - `model.joblib` (from `train.py`)

2. Optionally, place `incident_reports.csv` in `sensor_data/` for incident correlation analysis.

3. Verify directory structure:

   ```plaintext
   project/
   ├── app.py
   ├── generate_data.py
   ├── train.py
   ├── requirements.txt
   ├── results/
   │   ├── baseline_stats.csv
   │   ├── scaler.joblib
   │   ├── model.joblib
   ├── sensor_data/
   │   ├── training_data.csv (optional)
   │   ├── test_data.csv
   │   ├── incident_reports.csv
   ├── templates/
   │   ├── index.html
   │   ├── all_alerts.html
   └── static/ (for CSS/JS if needed)
   ```

## Running the Application

1. Ensure the MQTT broker (`broker.emqx.io`) is accessible and sensor data is being published to the specified topics.

2. Start the Flask application:

   ```bash
   python app.py
   ```

3. Access the dashboard:
   - Open a browser and navigate to `http://localhost:5000`
   - For remote access, use `http://<your-computer-ip>:5000` (e.g., `http://192.168.1.100:5000`).

4. The dashboard will:
   - Display real-time sensor readings (temperature, humidity, CO2)
   - Show up to three unacknowledged alerts (red for immediate hazards, yellow for trend-based)
   - Provide visualizations of sensor data, moving averages, baselines, and risk probabilities
   - Allow viewing all alerts at `/all-alerts`
   - Support acknowledging or deleting alerts via the UI

## Stopping the Application

- Press `Ctrl+C` in the terminal to stop the Flask server.
- The application will disconnect from the MQTT broker automatically.

## Troubleshooting

- **Data Generation (`generate_data.py`)**:
  - Ensure `sensor_data/` and `results/` directories exist.
  - Verify output files (`test_data.csv`, `incident_reports.csv`) are created.

- **Model Training (`train.py`)**:
  - Check that `training_data.csv` and `incident_reports.csv` are in `sensor_data/`.
  - Ensure data formats match (`;` separator, comma decimals for environmental data).
  - If training fails, verify sufficient risk labels (non-zero labels) in the data.

- **Flask App (`app.py`)**:
  - Confirm `results/` contains `baseline_stats.csv`, `scaler.joblib`, and `model.joblib`.
  - Ensure `templates/index.html` and `templates/all_alerts.html` exist.
  - If no data appears, verify MQTT topics are receiving data (`comp4436_gproj/sensor/*`).
  - Check Flask console logs for errors related to MQTT connection or model loading.
  - For port conflicts, edit `app.py` to change the port (e.g., `socketio.run(app, host='0.0.0.0', port=5001)`).

- **MQTT Issues**:
  - Ensure internet access to `broker.emqx.io:1883`.
  - Verify the external sensor system is publishing to the correct MQTT topics.
  - Use an MQTT client (e.g., `mosquitto_sub`) to test topic data:

     ```bash
     mosquitto_sub -h broker.emqx.io -t comp4436_gproj/sensor/#
     ```

## Notes

- The system assumes sensor data is published to `broker.emqx.io` at the specified topics. Ensure the external sensor system is operational.
- The Flask app uses a public MQTT broker for simplicity. For production, use a secure, private broker.
- Synthetic data from `generate_data.py` simulates realistic conditions but should be replaced with real historical data (`training_data.csv`) when available.
- The LogisticRegression model in `train.py` relies on incident-based labels. Ensure `incident_reports.csv` aligns with training data timelines.
- CO2 data accuracy depends on the external sensor’s calibration (e.g., if using an MQ-135, it’s sensitive to multiple gases like ammonia or alcohol).