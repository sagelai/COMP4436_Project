# Environmental Sensor Monitoring with MQTT

This project reads temperature, humidity, and CO2 levels using a DHT11 sensor and an ADC connected via I2C, then publishes the data to an MQTT broker.

## Prerequisites

- Raspberry Pi with Python 3
- DHT11 sensor connected to GPIO 26
- I2C ADC device at address 0x24
- Internet connection for MQTT communication

## Installation

1. Clone or download this project to your Raspberry Pi.

2. Navigate to the project directory:

   ```bash
   cd /path/to/project
   ```

3. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Hardware Setup

- Connect the DHT11 sensor to GPIO 26 (physical pin 37) on the Raspberry Pi.

- Connect the I2C ADC device to the I2C bus (SDA/SCL pins).

- Ensure I2C is enabled on your Raspberry Pi:

  ```bash
  sudo raspi-config
  ```

  Navigate to Interface Options &gt; I2C &gt; Enable

## Running the Program

1. Ensure your Raspberry Pi is connected to the internet.

2. Run the script:

   ```bash
   python sensor_mqtt.py
   ```

3. The program will:

   - Connect to the public EMQX MQTT broker
   - Read sensor data every few seconds
   - Publish temperature, humidity, and CO2 readings to MQTT topics:
     - comp4436_gproj/sensor/temperature
     - comp4436_gproj/sensor/humidity
     - comp4436_gproj/sensor/air_quality

## Stopping the Program

- Press `Ctrl+C` to stop the program gracefully.
- The script will disconnect from the MQTT broker and clean up resources.

## Troubleshooting

- If you see "Reading error" messages, check the DHT11 sensor connections.
- For I2C issues, verify the ADC address (0x24) and ensure I2C is enabled.
- If MQTT connection fails, verify internet connectivity and the broker address (broker.emqx.io).

## Notes

- The script uses a public MQTT broker for simplicity. For production, consider using a secure, private broker.
- The CO2 calculation assumes a specific sensor voltage range; adjust the `get_co2_ppm` function if using a different sensor.