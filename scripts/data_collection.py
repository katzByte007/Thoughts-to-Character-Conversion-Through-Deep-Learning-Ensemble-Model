import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import numpy as np

# Define ADS1115 parameters
ADS1115_SAMPLING_RATE = 256  # Sampling rate in Hz
NUM_SAMPLES = 50 

def collect_data():
    # Create the I2C bus
    i2c = busio.I2C(board.SCL, board.SDA)

    # Create the ADC object using the I2C bus
    ads = ADS.ADS1115(i2c)

    # Create analog input channel on pin A0
    channel = AnalogIn(ads, ADS.P0)

    samples = []
    for _ in range(NUM_SAMPLES):
        voltage = channel.voltage * 1000
        input_data = np.float64(voltage)
        samples.append(input_data)
        # Pause for the duration of each sample
        time.sleep(1 / ADS1115_SAMPLING_RATE)
    return np.array([samples])

    # return data_list
