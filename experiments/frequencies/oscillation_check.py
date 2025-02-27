import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from datetime import datetime
import math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

import os

sys.path.append("../..")
import snn
import tools


# Define the BRF update function
def brf_II_update(x, z_, u_, v_, q_, b_0, omega, dt=0.01, theta=0.1):
    b = b_0 - q_
    
    exp_bdt = torch.exp(b * dt)
    cos_omega_dt = torch.cos(omega * dt)
    sin_omega_dt = torch.sin(omega * dt)
    
    u_cos_sin = u_ * cos_omega_dt - v_ * sin_omega_dt
    v_cos_sin = u_ * sin_omega_dt + v_ * cos_omega_dt
    
    u = exp_bdt * u_cos_sin + x * dt
    v = exp_bdt * v_cos_sin
    q = 0.9 * q_ + z_
    
    z = (u - theta - q).gt(0).float()  # Simple thresholding
    
    return z, u, v, q

# Simulation parameters
T = 100 # Number of timesteps
dt = 0.01

#input
#x = torch.tensor(1.0)  # Constant input
omega_input = 30.0  # Input sine wave frequency
input_signal = torch.tensor(np.sin(omega_input * np.arange(T) * dt), dtype=torch.float32)


# Initialize state variables
u = torch.tensor(0.0)
v = torch.tensor(0.0)
q = torch.tensor(0.0)
z = torch.tensor(0.0)

# Model parameters
b_0 = torch.tensor(1.0)  # Default damping factor
omega = torch.tensor(30.0)  # Eigen angular frequency neuron
theta = 0.1  # Threshold

time = []
u_values = []
v_values = []

# Run simulation
for t in range(T):
    x=input_signal[t]
    z, u, v, q = brf_II_update(x, z, u, v, q, b_0, omega, dt, theta)
    time.append(t * dt)
    u_values.append(u.item())
    v_values.append(v.item())

# Plot results
plt.figure(figsize=(10, 4.5))
plt.plot(time, u_values, label='u (Real part)')
plt.plot(time, v_values, label='v (Imaginary part)')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Membrane Potential', fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.title('RF Neuron Response to sine wave input', fontsize=14)
plt.savefig("oscillations_check.png")
plt.close()


#############################
#BRF Cell
##############################

# Define input parameters
timesteps = 100
dt = 0.01
omega_value_input = 10.0  # Example angular frequency
omega_value_neuron = 10.0 
b_0 = torch.tensor(1.0) # Default b_offset
input_size = 1
layer_size = 1  # Single neuron for visualization

def sustain_osc(omega, dt: float = 0.01) -> torch.Tensor:
    omega = torch.tensor(omega) if not isinstance(omega, torch.Tensor) else omega
    return (-1 + torch.sqrt(1 - torch.square(dt * omega))) / dt


# Create a sine wave input
time = torch.arange(0, timesteps * dt, dt)
x_input = torch.sin(omega_value_input * time).reshape(timesteps, 1, 1)  # Shape [T, Batch, Features]

# Plot and save the input signal
plt.figure(figsize=(10, 5))
plt.plot(time.numpy(), x_input.squeeze().numpy(), label="Input Signal")
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Input Amplitude", fontsize=14)
plt.legend(fontsize=14, loc="lower left")
plt.title("Input Signal", fontsize=14)
plt.grid()
plt.savefig("input_signal.png")
plt.close()

# Initial state
z = torch.zeros(1, layer_size)
u = torch.zeros(1, layer_size)
v = torch.zeros(1, layer_size)
q = torch.zeros(1, layer_size)

p_omega = sustain_osc(omega_value_neuron)  # Use omega_value_neuron
b = p_omega - b_0 - q #b_0 default dump factor

state = (z, u, v, q)

# Store outputs
u_vals = []
v_vals = []

# Run the BRFCell over time
for t in range(timesteps):  # Ensure correct iteration over time
    x = x_input[t]
    z, u, v, q = brf_II_update(x, z, u, v, q, b, torch.tensor(omega_value_neuron), dt, theta)  # Use omega_value
    print(u.shape)
    u_vals.append(u.item())  # Append to the correct list
    v_vals.append(v.item())  # Append to the correct list

# Convert time back to NumPy array for plotting
time = time.numpy()

# Plot the oscillations
plt.figure(figsize=(10, 5))
plt.plot(time, u_vals, label="Real part (u)")
plt.plot(time, v_vals, label="Imaginary part (v)")
plt.xlabel("Time")
plt.ylabel("Membrane Potential")
plt.legend()
plt.title("Oscillations in BRFCell with Sine Wave Input")
plt.grid()
plt.savefig("cell_oscillations_check.png")
plt.close()
