import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os

# --- Configuration ---
DATA_DIR = 'sensor_data/'
RESULTS_DIR = 'results/'
# Output files FROM THIS SCRIPT
TEST_ENV_CSV_FILE = os.path.join(DATA_DIR, 'test_data.csv') # For monitoring later
ALIGNED_INC_CSV_FILE = os.path.join(DATA_DIR, 'incident_reports.csv') # For training

# --- Parameters for GENERATED Test Environmental Data ---
NUM_TEST_RECORDS = 1000 # Fewer records for test simulation
TEST_START_TIME = datetime(2023, 2, 1, 0, 0, 0, tzinfo=timezone.utc) # Start time for TEST data
TEST_TIME_INCREMENT_SECONDS = 30

# --- Parameters for GENERATED Incident Data (Aligned with your historical data) ---
# Your historical data spans roughly Jan 17 23:15 to Jan 22 19:06 UTC
# Let's generate incidents between Jan 18 01:00 and Jan 22 18:00 UTC
INCIDENT_START_TIME = datetime(2023, 1, 18, 1, 0, 0, tzinfo=timezone.utc)
INCIDENT_END_TIME = datetime(2023, 1, 22, 18, 0, 0, tzinfo=timezone.utc)
NUM_ALIGNED_INCIDENTS = 18 # Generate a reasonable number of incidents

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_test_environmental_data(num_records):
    """Generates environmental data for TESTING/MONITORING"""
    print(f"Generating {num_records} TEST environmental records...")
    data = []
    current_time = TEST_START_TIME
    temp = 17.0
    humidity = 62.3
    gas = 550

    for i in range(num_records):
        # Simulate some fluctuations and trends
        time_step = timedelta(seconds=np.random.normal(TEST_TIME_INCREMENT_SECONDS, 5))
        current_time += time_step

        temp += np.random.normal(0, 0.1) - 0.0001 * i # Slight downward trend
        humidity += np.random.normal(0, 0.01)
        # Ensure humidity stays relatively stable like historical data
        humidity = max(62.0, min(63.0, humidity + np.random.normal(0, 0.005)))
        gas += np.random.normal(0, 20) - 0.02 * i + np.random.choice([0, 150, 500, 1000], p=[0.97, 0.01, 0.01, 0.01]) # Random spikes like historical data

        # Clamp values to somewhat realistic ranges based on historical data
        temp = max(10.0, min(25.0, temp))
        # humidity clamp removed based on historical data pattern
        gas = max(250, min(3500, gas)) # Wider range based on historical spikes

        data.append({
            # Use standard ISO format with space separator + offset for test data
            'TIME': current_time.strftime('%Y-%m-%d %H:%M:%S%z'),
            'CO2': f'{gas:.0f}',
            'TEMPERATURE': f'{temp:.1f}'.replace('.', ','),
            'HUMIDITY': f'{humidity:.2f}'.replace('.', ',')
        })

    df = pd.DataFrame(data)
    print("Sample TEST environmental data:")
    print(df.head())
    return df

def generate_aligned_incident_data(num_incidents):
    """Generates incident data ALIGNED with the historical training data period"""
    print(f"\nGenerating {num_incidents} ALIGNED incident reports (Jan 18-22, 2023)...")
    incidents = []
    types = ['Gas Leak', 'Overheat', 'Maintenance', 'System Test', 'High CO2 Spike'] # Added type
    severities = ['Low', 'Medium', 'High', 'Critical'] # Added severity

    total_seconds_range = (INCIDENT_END_TIME - INCIDENT_START_TIME).total_seconds()

    for i in range(num_incidents):
        random_seconds = np.random.uniform(0, total_seconds_range)
        incident_time = INCIDENT_START_TIME + timedelta(seconds=random_seconds)

        # Correlate some incident types with potential data patterns from historical
        chosen_type = np.random.choice(types)
        if chosen_type in ['Gas Leak', 'High CO2 Spike']:
             chosen_severity = np.random.choice(['Medium', 'High', 'Critical'])
        elif chosen_type == 'Overheat':
             chosen_severity = np.random.choice(['Low', 'Medium','High'])
        else: # Maintenance, System Test
             chosen_severity = np.random.choice(['Low', 'Medium'])


        incidents.append({
            'incident_id': f'INC{i+1:03d}',
            # Use standard ISO format with space separator + offset
            'timestamp': incident_time.strftime('%Y-%m-%d %H:%M:%S%z'),
            'type': chosen_type,
            'severity': chosen_severity
        })

    df = pd.DataFrame(incidents)
    # Sort by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    # Convert back to string for saving
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

    print("Sample ALIGNED incident data:")
    print(df.head())
    return df

if __name__ == "__main__":
    print("--- Starting Data Generation (for Test Env Data & Aligned Incidents) ---")

    # 1. Generate Environmental Data for TESTING later
    test_env_df = generate_test_environmental_data(NUM_TEST_RECORDS)
    # Save with semicolon separator and comma decimal
    test_env_df.to_csv(TEST_ENV_CSV_FILE, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f"\nTEST Environmental data saved to {TEST_ENV_CSV_FILE}")

    # 2. Generate Incident Data aligned with HISTORICAL TRAINING data
    aligned_inc_df = generate_aligned_incident_data(NUM_ALIGNED_INCIDENTS)
    # Save with comma separator and standard decimal
    aligned_inc_df.to_csv(ALIGNED_INC_CSV_FILE, sep=',', decimal='.', index=False, encoding='utf-8-sig')
    print(f"\nALIGNED Incident data saved to {ALIGNED_INC_CSV_FILE}")

    print("\n--- Data Generation Finished ---")