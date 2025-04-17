# Construction Site Environmental Monitoring System

This is a web-based application for real-time environmental monitoring and safety alerts on construction sites. It displays environmental sensor data (temperature, humidity, CO₂ levels, gas concentration), an environmental safety index, and safety alerts, with a responsive UI built using Tailwind CSS, Chart.js, and Socket.io for real-time updates.

## Prerequisites

- **Python 3.8+**: Ensure Python is installed on your system.
- **pip**: Python package manager for installing dependencies.
- **Node.js** (optional): Only if you need to customize frontend dependencies.
- A modern web browser (e.g., Chrome, Firefox).

## Setup Instructions

Follow these steps to set up and run the application:

### 1. Clone the Repository
If you have a repository, clone it to your local machine. Otherwise, ensure you have the project files (`index.html`, `all_alerts.html`, `app.py`, etc.) in a project directory.

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
- `Flask`: Web framework for the backend.
- `Flask-SocketIO` and `python-socketio`: For real-time communication between server and client.
- `eventlet`: Asynchronous server for handling Socket.io connections.

### 4. Project Structure
Ensure your project directory has the following structure:
```
construction-site-monitoring/
├── static/
│   └── (frontend assets, if any, e.g., custom CSS/JS)
├── templates/
│   ├── index.html
│   └── all_alerts.html
├── app.py
├── requirements.txt
└── README.md
```

- `app.py`: The main Flask application file (must handle routes and Socket.io events).
- `templates/`: Contains HTML templates (`index.html`, `all_alerts.html`).
- `static/`: For custom static files (not required since Tailwind CSS, Chart.js, and Socket.io are loaded via CDNs).

### 5. Run the Application
Start the Flask application using the command below. Ensure you're in the project directory and the virtual environment is activated.

```bash
python app.py
```

By default, the application runs on `http://localhost:5000`. If `app.py` is configured differently, check its port settings.

### 6. Open the Application
Open a web browser and navigate to:

```
http://localhost:5000
```

You should see the dashboard with environmental monitoring data, the environmental safety index, and safety alerts. Click "View All Alerts" to access the alerts page.

### 7. Accessing Features
- **Environmental Monitoring**: View real-time sensor data (temperature, humidity, CO₂, gas concentration).
- **Environmental Safety Index**: Check the composite safety score and view details via the "View Details" button.
- **Safety Alerts**: See active alerts and manage them (acknowledge/delete) via popups.
- **All Alerts Page**: Access `/all-alerts` to filter and view all alerts by date, location, or severity.

## Troubleshooting
- **Port Conflict**: If `port 5000` is in use, modify `app.py` to use a different port (e.g., `app.run(port=5001)`).
- **Dependencies Fail**: Ensure `pip` is up-to-date (`pip install --upgrade pip`) and retry installing `requirements.txt`.
- **No Data Displayed**: Verify that `app.py` is correctly set up to emit Socket.io events (`sensor_update`, `env_index_update`, `alerts_update`).
- **UI Issues**: Ensure your browser supports modern JavaScript and has no ad-blockers interfering with CDN-loaded scripts (Tailwind CSS, Chart.js, Socket.io).

## Notes
- The frontend uses CDN-hosted libraries (Tailwind CSS, Chart.js, Socket.io, Flatpickr), so no Node.js setup is required unless you add custom frontend dependencies.
- Ensure `app.py` is implemented to handle Flask routes (`/`, `/all-alerts`) and Socket.io events for real-time updates.
- The application supports dark mode, toggled via the theme button in the UI.

For further assistance, contact the project maintainer or refer to the Flask and Flask-SocketIO documentation.
