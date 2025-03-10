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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = device.type == "cuda"
num_workers = 1 if device.type == "cuda" else 0
print(f"Using device: {device}")


#########################
#Generate dataset
#########################

class SineWaveDataset(Dataset):
    def __init__(self, omega_list, amplitude, phase_range, sequence_length, dt, num_samples, noise_std=0.05):
        self.omega_list = omega_list
        self.amplitude = amplitude
        self.phase_range = phase_range
        self.sequence_length = sequence_length
        self.dt = dt
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.data = []
        self.labels = []
        self._generate_dataset()

    def _generate_dataset(self):
        t = torch.arange(0, self.sequence_length * self.dt, self.dt)
        for label, omega in enumerate(self.omega_list):
            for _ in range(self.num_samples // len(self.omega_list)):
                phase = random.uniform(*self.phase_range)
                sine_wave = self.amplitude * torch.sin(omega * t + phase) 
                noise = torch.normal(0, self.noise_std, size=sine_wave.size())
                noisy_wave = sine_wave + noise
                self.data.append(noisy_wave.view(self.sequence_length, 1, 1))  # Reshape to [T, 1, 1]
                self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

################################################################
# Data preparation
################################################################

rand_num = random.randint(1, 10000)
omega_list = [10.0, 17.0, 25.0]
num_classes = len(omega_list)
omega_list_neurons = [10.0, 17.0, 25.0]
amplitude = 1
phase_range = (0, 0) # No phase shift
sequence_length = 200
dt = 0.01
num_samples = 3000
noise_std = 0.00
batch_size = 256

dataset = SineWaveDataset(
    omega_list=omega_list,
    amplitude=amplitude,
    phase_range=phase_range,
    sequence_length=sequence_length,
    dt=dt,
    num_samples=num_samples,
    noise_std=noise_std
)

# Example of correct shape check
#sample_data, _ = dataset[0]
#print("Sample shape:", sample_data.shape)  # Should print [100, 1, 1]


def get_random_sample_per_class(dataset, num_classes):
    class_samples = {}
    for i in range(len(dataset)):
        data, label = dataset[i]
        if label.item() not in class_samples:
            class_samples[label.item()] = (data, label.item())
        if len(class_samples) == num_classes:
            break
    return class_samples

# Example usage
class_samples = get_random_sample_per_class(dataset, num_classes)

# Prepare inputs and labels
inputs = torch.stack([class_samples[i][0] for i in range(num_classes)])
labels = torch.tensor([class_samples[i][1] for i in range(num_classes)])

# Check shapes
#print(inputs.shape)  # Should print torch.Size([num_classes, sequence_length, 1, 1])
#print(labels.shape)  # Should print torch.Size([num_classes])


def plot_samples(dataset, class_samples, num_classes, save_path="input_waves.png"):
    # Extract time and amplitude
    t = torch.arange(0, dataset.sequence_length * dataset.dt, dataset.dt)

    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot each sample for every class
    for label in range(num_classes):
        data = class_samples[label][0]
        plt.plot(t.numpy(), data.numpy().squeeze(), label=f"Class {label} (omega={dataset.omega_list[label]})")
    
    # Adding labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Sine Wave Samples with Noise")
    plt.legend()

    # Save the plot as an image
    plt.savefig(save_path)
    plt.close()

# Example usage
plot_samples(dataset, class_samples, num_classes)



# Define the BRF update function
def brf_II_update(x, z_, u_, v_, q_, b_0, omega, dt=0.01, theta=0.1, gamma=0.8):
    b = b_0 - q_
    
    exp_bdt = torch.exp(b * dt)
    cos_omega_dt = torch.cos(omega * dt)
    sin_omega_dt = torch.sin(omega * dt)
    
    u_cos_sin = u_ * cos_omega_dt - v_ * sin_omega_dt
    v_cos_sin = u_ * sin_omega_dt + v_ * cos_omega_dt
    
    u = exp_bdt * u_cos_sin + x * dt
    v = exp_bdt * v_cos_sin
    q = gamma * q_ + z_
    
    z = (u - theta - q).gt(0).float()  # Simple thresholding
    
    return z, u, v, q

def sustain_osc(omega, dt: float = 0.01) -> torch.Tensor:
    omega = torch.tensor(omega) if not isinstance(omega, torch.Tensor) else omega
    return (-1 + torch.sqrt(1 - torch.square(dt * omega))) / dt


def run_brf_for_samples(inputs, num_classes, omega_list_neurons, timesteps=sequence_length, dt=0.01, theta=0.1, b_0=1.0):
    num_neurons = len(omega_list_neurons)  # Number of neurons
    
    for i in range(num_classes):
        x_input = inputs[i]  # Get input sample for current class
        #print(x_input.shape) [time_steps, 1, 1] (time, batch , features)
        # Initial states for all neurons
        z = torch.zeros(1,num_neurons)
        u = torch.zeros(1,num_neurons)
        v = torch.zeros(1,num_neurons)
        q = torch.zeros(1,num_neurons)

        p_omega = torch.tensor([sustain_osc(omega) for omega in omega_list_neurons])
        b = p_omega - b_0 - q  # Default damping factor
        #print(b.shape) correct
        # Store outputs
        u_vals = [[] for _ in range(num_neurons)]
        z_vals = [[] for _ in range(num_neurons)]
        spike_times = []
        spike_neuron_indices = []
        #print(f"u shape: {u.shape}, v shape: {v.shape}, q shape: {q.shape}, z shape: {z.shape}")
        # Run the BRF cell over time
        for t in range(timesteps):
            x = x_input[t]  # Current time step input
            #print(x.shape) [1,1]
            z, u, v, q = brf_II_update(x, z, u, v, q, b, torch.tensor(omega_list_neurons), dt, theta)
            #print(f"u shape: {u.shape}, v shape: {v.shape}, q shape: {q.shape}, z shape: {z.shape}")
            u = u.squeeze(0)  # Shape becomes [3]
            z = z.squeeze(0)
            for n in range(num_neurons):
                u_vals[n].append(u[n].detach().cpu().numpy())  # Store u values
                z_vals[n].append(z[n].detach().cpu().numpy())  # Store z values

                #print(z[n])
                if (z[n].detach().cpu().numpy() > 0).any():
                #if z[n].gt(0): # Record spikes for raster plot
                    spike_times.append(t * dt)
                    spike_neuron_indices.append(n)

        # Time axis for plotting
        time = torch.arange(0, timesteps * dt, dt).numpy()
        
        # Plot u values over time for all neurons
        plt.figure(figsize=(10, 6))
        for n in range(num_neurons):
            plt.plot(time, u_vals[n], label=f"Neuron {n+1} (omega={omega_list_neurons[n]})")
        plt.xlabel("Time (s)")
        plt.ylabel("Membrane Potential (u)")
        plt.legend(loc="best")
        plt.title(f"Membrane Potential Over Time - Class {i+1}")
        plt.grid(True)
        plt.savefig(f"brf_u_vals_class_{i+1}.png")
        plt.close()

        # Raster plot of spikes
        plt.figure(figsize=(10, 6))
        plt.scatter(spike_times, spike_neuron_indices, marker="|", color="black", s=100)
        plt.xlabel("Time (s)")
        plt.ylabel("Neuron Index")
        plt.title(f"Raster Plot - Class {i+1}")
        plt.yticks(range(num_neurons), [f"Neuron {n+1}" for n in range(num_neurons)])
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f"brf_raster_plot_class_{i+1}.png")
        plt.close()

#print(inputs.shape)
# Example of running with multiple neuron # Three neurons with different frequencies
run_brf_for_samples(inputs, num_classes, omega_list_neurons)
'''
##########################
#Model and simulation
#########################`

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


def sustain_osc(omega, dt: float = 0.01) -> torch.Tensor:
    omega = torch.tensor(omega) if not isinstance(omega, torch.Tensor) else omega
    return (-1 + torch.sqrt(1 - torch.square(dt * omega))) / dt



def run_brf_for_samples(inputs, num_classes, timesteps=100, dt=0.01, omega_value_neuron=10.0, theta=0.1, b_0=1.0):
    # Loop through each sample and plot independently
    for i in range(num_classes):
        x_input = inputs[i]  # Get input sample for current class
        # Initial state
        layer_size=1
        z = torch.zeros(1, layer_size)
        u = torch.zeros(1, layer_size)
        v = torch.zeros(1, layer_size)
        q = torch.zeros(1, layer_size)

        p_omega = sustain_osc(omega_value_neuron)  # Use omega_value_neuron
        b = p_omega - b_0 - q  # b_0 default dump factor

        # Store outputs
        u_vals = []
        v_vals = []

        # Run the BRF cell over time
        for t in range(timesteps):
            x = x_input[t]  # Current time step input
            z, u, v, q = brf_II_update(x, z, u, v, q, b, torch.tensor(omega_value_neuron), dt, theta)
            u_vals.append(u.item())  # Append u values
            v_vals.append(v.item())  # Append v values
        
        # Plot each sample's u and v values over time
        time = torch.arange(0, timesteps * dt, dt).numpy()  # Time axis
        
        # Create a new figure for each sample, showing both u and v
        plt.figure(figsize=(10, 6))
        plt.plot(time, u_vals, label="u (Real part)")
        plt.plot(time, v_vals, label="v (Imaginary part)")
        plt.xlabel("Time")
        plt.ylabel("Membrane Potential")
        plt.legend(loc="best")
        plt.title(f"Oscillations in BRF Cell - Class {i+1}")
        plt.grid(True)

        # Save or show the figure
        plt.savefig(f"ripro_oscillations_class_{i+1}.png")  # Save with unique name per class
        plt.close() 

# Run simulation and plot
run_brf_for_samples(inputs, num_classes)


##########################So far it works as expected
'''

