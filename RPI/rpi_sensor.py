import board
import adafruit_dht
import time
import paho.mqtt.client as mqtt
import random
import smbus

bus = smbus.SMBus(1)
I2C_ADDRESS = 0x24

# Initialize DHT11 sensor
dht = adafruit_dht.DHT11(board.D26)  # GPIO 26

# MQTT setup
mqtt_broker = "broker.emqx.io"
mqtt_port = 1883
mqtt_topic_temp = "comp4436_gproj/sensor/temperature"
mqtt_topic_humid = "comp4436_gproj/sensor/humidity"
mqtt_topic_air = "comp4436_gproj/sensor/air_quality"

def read_adc(channel_reg):
	bus.write_byte(I2C_ADDRESS, channel_reg)
	data = bus.read_word_data(I2C_ADDRESS, channel_reg)
	return data

def publish_with_verification(client, topic, payload=None, qos=1):
    """Publish with verification and detailed logging"""
    print(f"Attempting to publish to topic '{topic}'")
    print(f"Payload: {payload}")
    
    try:
        # Publish and get the result
        result = client.publish(topic, payload, retain=True, qos=qos)
        
        # Check the return code
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"? Publication to '{topic}' queued successfully")
            # Wait for message to be sent
            result.wait_for_publish()
            if result.is_published():
                print(f"? Publication to '{topic}' confirmed")
            else:
                print(f"? Publication to '{topic}' failed after queuing")
        else:
            print(f"? Publication to '{topic}' failed with code {result.rc}")
    except Exception as e:
        print(f"? Exception during publication to '{topic}': {e}")

def get_co2_ppm(sensorValue):
    voltage = sensorValue * (5000/1024.0)
    if voltage == 0:
        return 0
    elif voltage < 400:
        print("Preheating")
        return 0
    
    voltage_diff = voltage-400
    ppm = voltage_diff*50.0/16.0
    return ppm

try:
    # Connect to MQTT broker
    print(f"Connecting to MQTT broker at {mqtt_broker}:{mqtt_port}...")
    client = mqtt.Client()
    client.connect(mqtt_broker, mqtt_port)
    client.loop_start()
    print("Connected to MQTT broker")
    
    print("Starting monitoring loop...")
    while True:
        try:
            # Read temperature and humidity
            temperature = dht.temperature
            humidity = dht.humidity
            co2 = get_co2_ppm(read_adc(0x10))
            
            print(f'Temperature: {temperature:.1f} Humidity: {humidity:.1f}% CO2: {co2:.1f} PPM')
            
            # Publish to MQTT
            try:
                publish_with_verification(client, mqtt_topic_temp, f"{temperature:.1f}")
                publish_with_verification(client, mqtt_topic_humid, f"{humidity:.1f}")
                publish_with_verification(client, mqtt_topic_air, f"{co2:.1f}")
                print("Published data to MQTT topics")
            except Exception as e:
                print(f'Error publishing to MQTT: {e}')
                
        except RuntimeError as e:
            # Reading can fail - DHT11 sometimes gives bad data
            print(f'Reading error: {e}')
        
except KeyboardInterrupt:
    print("Program stopped by user")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # Clean up
    print("Cleaning up...")
    try:
        client.loop_stop()
        client.disconnect()
        print("Disconnected from MQTT broker")
    except:
        pass
    dht.exit()
    print("Program ended")
