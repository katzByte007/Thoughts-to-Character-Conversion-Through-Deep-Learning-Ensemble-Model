import random
import time
import numpy as np
import torch

# Define parameters
ADS1115_SAMPLING_RATE = 256  # Sampling rate in Hz
DURATION = 2  # Duration in seconds for each recording
NUM_SAMPLES = 50  # Number of samples

def generate_random_input():
    samples = []
    for _ in range(NUM_SAMPLES):
        # Generate a random number between 1700 and 3300
        input_data = np.random.random() * (3000.0 - 1700.0) + 1700.0
        input_data = np.float64(input_data)
        samples.append(input_data)
        # Pause for the duration of each sample
        time.sleep(1 / ADS1115_SAMPLING_RATE)
    return np.array([samples])

if __name__ == "__main__":
    generate_random_input()
