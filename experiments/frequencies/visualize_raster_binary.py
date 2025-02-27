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


class BinarySineDataset(Dataset):
    def __init__(self, omega_list, dt, sequence_length, num_samples):
        self.omega_list = omega_list
        self.dt = dt
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        self._generate_dataset()

    def _generate_dataset(self):
        for label, omega in enumerate(self.omega_list):
            period = (2 * math.pi) / omega  # Compute the period of the sine wave
            step_interval = max(1, round(period / self.dt))  # Convert to discrete steps
            
            for _ in range(self.num_samples // len(self.omega_list)):
                sequence = torch.zeros(self.sequence_length)
                index = 0  # Start placing '1's from the beginning
                while index < self.sequence_length:
                    sequence[index] = 50  # Place a spike
                    index += step_interval  # Move forward by step_interval
                
                self.data.append(sequence.unsqueeze(-1))  # Add feature dimension
                self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Parameters
omega_list = [10.0, 22.0, 46.0]  # Angular frequencies
sequence_length = 100
num_samples = 3000
dt = 0.01
batch_size = 256

# Create dataset
binary_sine_dataset = BinarySineDataset(
    omega_list=omega_list, dt=dt, sequence_length=sequence_length, num_samples=num_samples
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
#omega_list = [10.0, 27.0, 78.0]
dt = 0.01
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

def save_binary_sequences(inputs, labels, omega_list, dt):
    """
    Save a raster plot of the binary sequences with time on the x-axis.
    
    inputs: Tensor of shape (num_samples, sequence_length, 1)
    labels: Tensor of shape (num_samples,)
    omega_list: List of omega values
    dt: Time step used for sampling the binary sequences
    """

    num_samples, sequence_length, _ = inputs.shape
    time_axis = np.arange(0, sequence_length * dt, dt)

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        # Plot spikes as vertical lines at each 1 in the binary sequence
        spike_times = time_axis[inputs[i, :, 0].cpu().numpy() !=0]
        plt.scatter(spike_times, np.ones_like(spike_times) * i, c='black', marker='|', s=50)

    plt.xlabel("Time (s)")
    plt.ylabel("Sample Index")
    plt.title("Binary Sequence Raster Plot")
    plt.yticks(np.arange(0, num_samples, step=1), labels=np.arange(0, num_samples, step=1))
    plt.grid(True)
    plt.tight_layout()
    #plt.legend()
    plt.savefig("binary_sequence_raster_plot.png")
    plt.close()

# Call the function to save the raster plot
save_binary_sequences(inputs, labels, omega_list, dt)
print("Binary sequence raster plot saved as 'binary_sequence_raster_plot.png'")


# Function to save raster plot as an image

def save_raster_plot(spikes, labels, omega_list, save_path="raster_plot_binary.png"):
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
        #ax.legend()
        # Save each figure separately
        plt.tight_layout()
        #ax.legend()
        plt.savefig(f"{save_path}_sample{sample_idx}.png")
        plt.close()

def save_hidden_u_plot(hidden_values, labels, omega_list, save_path="hidden_u_plot_binary.png"):
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
        ax.legend()
        # Save each figure separately
        plt.tight_layout()
        #plt.legend()
        plt.savefig(f"{save_path}_sample{sample_idx}.png")
        plt.close()


# Pass through model one sample at a time
hidden_spikes_list = []
hidden_u_list=[]

with torch.no_grad():
    for i in range(inputs.shape[0]): 

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
save_hidden_u_plot(hidden_u, labels, omega_list, save_path="hidden_u_plot_binary.png")
print("Raster plot saved as 'raster_plots_binary.png'")

