## Prerequisites

- Raspberry Pi (2B/3B/3B+/4/Zero) with Python 3
- [Emakefun RaspberryPi-Sensor-Board](https://github.com/emakefun/RaspberryPi-Sensor-Board) for ADC functionality 
- DHT11 sensor connected to GPIO 26
- MQ-135 gas sensor connected to the Emakefun sensor board’s ADC channel
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

- Connect the Emakefun RaspberryPi-Sensor-Board to the Raspberry Pi

- Connect the DHT11 sensor to GPIO 26 and MQ-135 sensor to A0 on the Emakefun RaspberryPi-Sensor-Board.

- Enable I2C on your Raspberry Pi:

  ```bash
  sudo raspi-config
  ```

  Navigate to Interface Options > I2C > Enable

- Verify the sensor board is detected at I2C address 0x24:

  ```bash
  sudo i2cdetect -y 1
  ```

## Running the Program

1. Ensure your Raspberry Pi is connected to the internet.

2. Preheat the MQ-135 sensor for 24–48 hours in clean air to stabilize its readings (recommended for first use or after long storage).

3. Run the script:

   ```bash
   python rpi_sensor.py
   ```

## Stopping the Program

- Press `Ctrl+C` to stop the program gracefully.
- The script will disconnect from the MQTT broker and clean up resources.

## Troubleshooting

- **DHT11 Reading Errors**: Check sensor connections, power (3.3V or 5V), and pull-up resistor. Retry after a few seconds, as DHT11 can be temperamental.
- **I2C/ADC Issues**: Confirm the Emakefun sensor board is at address 0x24 using `i2cdetect`. Ensure I2C is enabled and `smbus2` is installed.
- **MQ-135 Issues**:
  - If readings are erratic, ensure the sensor is preheated (24–48 hours for initial use).
  - Verify 5V power supply and correct ADC channel (0x10).
  - Check for interference from other gases (e.g., alcohol, ammonia), as MQ-135 is not CO2-specific.
- **MQTT Connection Issues**: Verify internet connectivity and the broker address (broker.emqx.io).

## Notes

- The Emakefun RaspberryPi-Sensor-Board enables ADC functionality for the MQ-135’s analog output, as the Raspberry Pi cannot read analog signals directly [Ref: Emakefun RaspberryPi-Sensor-Board](https://github.com/emakefun/RaspberryPi-Sensor-Board).
- The MQ-135 is sensitive to multiple gases (CO2, ammonia, alcohol, etc.), so readings reflect general air quality, not CO2 alone. The `get_co2_ppm` function assumes a linear voltage-to-ppm conversion (400mV baseline, 50ppm per 16mV), which is a simplification. For accurate CO2 measurements:
  - Calibrate in clean air (400 ppm CO2 baseline).
  - Adjust calculations based on the MQ-135 datasheet and load resistor value (typically 10kΩ–47kΩ).
  - Consider environmental factors like temperature and humidity (read from DHT11) for compensation.
- The script uses a public MQTT broker for simplicity. For production, use a secure, private broker.
- MQ-135 requires a stable 5V power source for its heater. Ensure your power supply can handle the current draw.