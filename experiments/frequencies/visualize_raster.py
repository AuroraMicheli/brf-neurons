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

################################################################
# General settings
################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = device.type == "cuda"
num_workers = 1 if device.type == "cuda" else 0
print(f"Using device: {device}")

################################################################
# Custom dataset for sine wave classification
################################################################
'''''
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
                self.data.append(noisy_wave.unsqueeze(-1))  # Add feature dimension
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
#omega_list = [10.0, 26.0, 57.0]
omega_list = [10.0, 29.0, 65.0]
amplitude = 1.0
#phase_range = (0, 2 * math.pi)
phase_range = (0, 0)
sequence_length = 1000
dt = 0.01
num_samples = 3000
noise_std = 0.00
batch_size = 256

sine_dataset = SineWaveDataset(
    omega_list=omega_list,
    amplitude=amplitude,
    phase_range=phase_range,
    sequence_length=sequence_length,
    dt=dt,
    num_samples=num_samples,
    noise_std=noise_std
)

# Split into train, validation, and test sets (70/15/15 split)
train_size = int(0.7 * len(sine_dataset))
val_size = int(0.15 * len(sine_dataset))
test_size = len(sine_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(sine_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
'''
################################################################
# Custom dataset for binary sine wave classification
################################################################
class BinarySineWaveDataset(Dataset):
    def __init__(self, omega_list, sequence_length, dt, num_samples, phase_range=(0, 0)):
        self.omega_list = omega_list
        self.sequence_length = sequence_length
        self.dt = dt
        self.num_samples = num_samples
        self.phase_range = phase_range
        self.data = []
        self.labels = []
        self._generate_dataset()

    def _generate_dataset(self):
        t = torch.arange(0, self.sequence_length * self.dt, self.dt)
        for label, omega in enumerate(self.omega_list):
            for _ in range(self.num_samples // len(self.omega_list)):
                phase = random.uniform(*self.phase_range)
                sine_wave = torch.sin(omega * t + phase)
                binary_wave = (sine_wave >= 0.5).float()  # Convert to binary (0 or 1)
                self.data.append(binary_wave.unsqueeze(-1))  # Add feature dimension
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
omega_list = [10.0, 29.0, 65.0]
sequence_length = 100
dt = 0.01
num_samples = 3000
batch_size = 256

binary_sine_dataset = BinarySineWaveDataset(
    omega_list=omega_list,
    sequence_length=sequence_length,
    dt=dt,
    num_samples=num_samples
)

# Split into train, validation, and test sets (70/15/15 split)
train_size = int(0.7 * len(binary_sine_dataset))
val_size = int(0.15 * len(binary_sine_dataset))
test_size = len(binary_sine_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(binary_sine_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model setup
hidden_size = 1
num_classes = len(omega_list)
input_size = 1  # Single feature per time step

omega_list_neurons = [10.0]
# Define your SNN model (update this import if necessary)
model = snn.models.SimpleResRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    #adaptive_omega_a=15.0,
    #adaptive_omega_b=85.0,
    adaptive_omega_a=min(omega_list)- 0.5,
    adaptive_omega_b=max(omega_list)+0.5,
    adaptive_b_offset_a=0.1,
    adaptive_b_offset_b=1.0,
    out_adaptive_tau_mem_mean=20.0,
    out_adaptive_tau_mem_std=1.0,
    label_last=False,
    mask_prob=0.0,
    output_bias=False,
    initial_omegas=omega_list_neurons
).to(device)

model = torch.jit.script(model)

# Function to get random sample per class and store their respective class (omega)
def get_random_sample_per_class(dataset, num_classes):
    class_samples = {}
    for i in range(len(dataset)):
        data, label = dataset[i]
        if label.item() not in class_samples:
            class_samples[label.item()] = (data, label.item())
        if len(class_samples) == num_classes:
            break
    return class_samples

# Get one sample for each class
class_samples = get_random_sample_per_class(test_dataset, num_classes)

# Prepare inputs and labels
inputs = torch.stack([class_samples[i][0] for i in range(num_classes)]).to(device)
#print(inputs[0,:,:])
labels = torch.tensor([class_samples[i][1] for i in range(num_classes)]).to(device)

################################################################
# Function to save input sine waveforms
################################################################

def save_input_waveforms(inputs, labels, omega_list, dt):
    """
    Save a plot of the three input sine waves.
    
    inputs: Tensor of shape (num_classes, sequence_length, 1)
    labels: Tensor of shape (num_classes,)
    omega_list: List of omega values
    dt: Time step used for sampling the sine waves
    """

    num_classes, sequence_length, _ = inputs.shape
    time_axis = np.arange(0, sequence_length * dt, dt)

    plt.figure(figsize=(12, 4))
    for i in range(num_classes):
        plt.plot(time_axis, inputs[i, :, 0].cpu().numpy(), label=f"Class {labels[i].item()} (Omega: {omega_list[labels[i].item()]})")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Input Sine Waveforms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("input_waveforms.png")
    plt.close()

# Call the function to save the plot
save_input_waveforms(inputs, labels, omega_list, dt)
print("Input waveforms saved as 'input_waveforms.png'")


# Function to save raster plot as an image

def save_raster_plot(spikes, labels, omega_list, save_path="raster_plot.png"):
    """
    Save separate raster plots for each sample/class in the dataset.
    
    spikes: numpy array with shape (num_samples, time_steps, num_neurons), binary (0/1).
    labels: class labels for each sample.
    omega_list: list of omega values associated with the classes.
    save_path: file path prefix for saving raster plots.
    """
    num_samples, time_steps, num_neurons = spikes.shape  # (S, T, N)

    for sample_idx in range(num_samples):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Loop through neurons and plot spikes as dots
        for neuron_idx in range(num_neurons):
            spike_times = np.where(spikes[sample_idx, :, neuron_idx] > 0)[0]  # Get time indices where spikes occur
            ax.scatter(spike_times, np.full_like(spike_times, neuron_idx), 
                       s=20, c='black', marker='x', label=f"Neuron {neuron_idx}" if sample_idx == 0 else None)

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Neurons")
        ax.set_yticks(range(num_neurons))
        ax.set_yticklabels([f"Neuron {i}" for i in range(num_neurons)])
        ax.set_title(f"Sample {sample_idx} - Class {labels[sample_idx].item()} - Omega: {omega_list[labels[sample_idx].item()]}")

        # Save each figure separately
        plt.tight_layout()
        plt.savefig(f"{save_path}_sample{sample_idx}.png")
        plt.close()

def save_hidden_u_plot(hidden_values, labels, omega_list, save_path="hidden_u_plot.png"):
    """
    Save a plot of the hidden state values over time for each class.
    
    hidden_values: numpy array with shape (num_samples, time_steps), real-valued.
    labels: class labels for each sample.
    omega_list: list of omega values associated with the classes.
    save_path: file path prefix for saving hidden state plots.
    """
   
    num_samples, time_steps, num_neurons = hidden_values.shape
    time_axis = np.arange(time_steps)

    for sample_idx in range(num_samples):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Loop through neurons and plot spikes as dots
        for neuron_idx in range(num_neurons):
            ax.plot(time_axis, hidden_values[sample_idx, :, neuron_idx], label=f"Neuron {neuron_idx}")

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("U values")
        #ax.set_yticks(range(num_neurons))
        #ax.set_yticklabels([f"Neuron {i}" for i in range(num_neurons)])
        #ax.set_title(f"Sample {sample_idx} - Class {labels[sample_idx].item()} - Omega: {omega_list[labels[sample_idx].item()]}")

        # Save each figure separately
        plt.tight_layout()
        plt.savefig(f"{save_path}_sample{sample_idx}.png")
        plt.close()


# Pass through model one sample at a time
hidden_spikes_list = []
hidden_u_list=[]

with torch.no_grad():
    for i in range(inputs.shape[0]): 

        model = torch.jit.script(model) # Loop over each sample
        #print(inputs.shape)
        input_sample = inputs[i].unsqueeze(0)  # Add batch dimension -> [1, 1000, 1]
        output, (hidden_states, out_u), num_spikes = model(input_sample)
        
        hidden_z = hidden_states[0]  # Extract hidden spikes
        hidden_u = hidden_states[1]
        #print(hidden_u.shape)
        # Ensure shape is [time_steps, num_neurons]
        hidden_spikes_list.append(hidden_z.squeeze(0).cpu().numpy())  
        hidden_u_list.append(hidden_u.squeeze(0).cpu().numpy())

# Stack to get shape (num_samples, time_steps, num_neurons)
hidden_spikes = np.stack(hidden_spikes_list, axis=0)
hidden_u = np.stack(hidden_u_list, axis=0)

# Save raster plot for each sample
save_raster_plot(hidden_spikes, labels, omega_list)
save_hidden_u_plot(hidden_u, labels, omega_list, save_path="hidden_u_plot.png")
print("Raster plot saved as 'raster_plots.png'")

