import json
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta

# Define the function to calculate safety index (updated to include MQ2 gas concentration)
def calculate_safety_index(temp, humidity, air_quality, gas_concentration):
    # Temperature score
    temp_score = max(0, 100 - abs(temp - 25) * 10)
    if temp > 40:
        temp_score = 0

    # Humidity score
    humidity_score = max(0, 100 - abs(humidity - 55) * 5)
    if humidity > 90:
        humidity_score = 0

    # Air quality score
    air_score = max(0, 100 - (air_quality - 50) // 10 * 10)
    if air_quality > 200:
        air_score = 0

    # Gas concentration score (MQ2 sensor, assuming ppm)
    # Assuming safe range is 0-200 ppm, above 200 ppm starts reducing score
    gas_score = max(0, 100 - (max(0, gas_concentration - 200) // 10 * 5))
    if gas_concentration > 1000:  # Critical threshold
        gas_score = 0

    # Weighted average (adjust weights to include gas concentration)
    safety_index = (temp_score * 0.25 + humidity_score * 0.25 + air_score * 0.25 + gas_score * 0.25)
    return safety_index

# Simulated incident timestamps
INCIDENTS = ["2025-04-07 10:00:00", "2025-04-07 10:10:00"]

# Generate both environmental_data and incidents, including MQ2 sensor data
def generate_combined_data_to_json():
    # Step 1: Generate environmental_data with MQ2 sensor data
    base_time = datetime.datetime(2025, 4, 7, 8, 0, 0)
    np.random.seed(42)  # Set random seed for reproducibility

    env_data = {}

    for i in range(100):
        timestamp = (base_time + timedelta(minutes=2 * i))
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        progress = i / 100
        temp = np.random.normal(25 + 5 * progress, 2)  # 25°C to 30°C
        humidity = np.random.normal(55 + 10 * progress, 3)  # 55% to 65%
        air_quality = np.random.normal(50 + 50 * progress, 10)  # 50 to 100 AQI
        # MQ2 gas concentration (ppm), increasing over time to simulate potential hazard
        gas_concentration = np.random.normal(150 + 300 * progress, 20)  # 150 to 450 ppm
        safety_index = calculate_safety_index(temp, humidity, air_quality, gas_concentration)

        env_data[timestamp_str] = {
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'air_quality': round(air_quality, 2),
            'gas_concentration': round(gas_concentration, 2),  # Add MQ2 sensor data
            'safety_index': round(safety_index, 2)
        }

    # Step 2: Generate incidents based on environmental_data
    incidents_data = {}

    for incident in INCIDENTS:
        incident_time = pd.to_datetime(incident, utc=True)

        closest_data = None
        min_time_diff = float('inf')

        for timestamp, data in env_data.items():
            env_timestamp = pd.to_datetime(timestamp, utc=True)
            time_diff = abs((env_timestamp - incident_time).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_data = data

        if closest_data:
            safety_index = closest_data['safety_index']
            gas_concentration = closest_data['gas_concentration']
            # Adjust severity based on safety index and gas concentration
            if safety_index < 30 or gas_concentration > 800:  # High severity if gas is very high
                severity = "High"
                description = "Critical environmental conditions detected (High gas concentration)" if gas_concentration > 800 else "Critical environmental conditions detected"
            elif safety_index < 50 or gas_concentration > 500:  # Medium severity if gas is moderately high
                severity = "Medium"
                description = "Elevated environmental conditions (Elevated gas concentration)" if gas_concentration > 500 else "Elevated environmental conditions"
            else:
                severity = "Low"
                description = "Minor environmental concerns"

            timestamp_str = incident_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            incidents_data[timestamp_str] = {
                'description': description,
                'severity': severity,
                'related_data': {
                    'temperature': closest_data['temperature'],
                    'humidity': closest_data['humidity'],
                    'air_quality': closest_data['air_quality'],
                    'gas_concentration': closest_data['gas_concentration'],  # Include MQ2 data
                    'safety_index': safety_index
                }
            }

    # Step 3: Combine both datasets into a single dictionary
    combined_data = {
        'environmental_data': env_data,
        'incidents': incidents_data
    }

    # Step 4: Save to a single JSON file
    with open('combined_data.json', 'w') as f:
        json.dump(combined_data, f, indent=4)
    print("Generated and saved combined data (environmental_data and incidents) to combined_data.json")

if __name__ == "__main__":
    generate_combined_data_to_json()