import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
import math
import random
import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
#print(matplotlib.__version__)
import wandb
import os
sys.path.append("../..")
import snn
import tools

seed=42 #with seed 42, acc=100 after 24 epochs. with seed 50, acc=30 after >40 epochs
random.seed(seed)
torch.manual_seed(seed)
numpy.random.seed(seed)

################################################################
# General settings
################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = device.type == "cuda"
num_workers = 1 if device.type == "cuda" else 0
print(f"Using device: {device}")

################################################################
# Custom dataset for pulses classification
################################################################

class DeltaPulseDataset(Dataset):
    def __init__(self, omega_list, amplitude, phase_range, sequence_length, dt, num_samples, noise_std=0.00, noise_value=0.1):
        self.omega_list = omega_list
        self.amplitude = amplitude
        self.phase_range = phase_range
        self.sequence_length = sequence_length
        self.dt = dt
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.noise_value = noise_value
        self.data = []
        self.labels = []
        self._generate_dataset()

    def _generate_dataset(self):
        t = torch.arange(0, self.sequence_length * self.dt, self.dt)
        
        for label, omega in enumerate(self.omega_list):
            for _ in range(self.num_samples // len(self.omega_list)):
                phase = random.uniform(*self.phase_range)
                sine_wave = self.amplitude * torch.sin(omega * t + phase)
                
                # Generate delta pulses: Impulse at sine wave peaks
                delta_pulses = torch.normal(0, self.noise_value, size=sine_wave.size())
                peaks = (sine_wave[1:-1] > sine_wave[:-2]) & (sine_wave[1:-1] > sine_wave[2:])
                delta_pulses[1:-1][peaks] = 5.0  # Set delta pulses at peaks
                
                # Add noise if needed
                noise = torch.normal(0, self.noise_std, size=delta_pulses.size())
                noisy_pulses = delta_pulses + noise
                
                self.data.append(noisy_pulses.unsqueeze(-1))  # Add feature dimension
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

# Data preparation
omega_list = [10.0, 25.0, 56.0]
omega_list_neurons = omega_list
amplitude = 1.0
phase_range = (0, 0)
sequence_length = 500
dt = 0.01
num_samples = 3000
noise_std = 0.01
noise_value = 0.1
batch_size = 256

dataset = DeltaPulseDataset(
    omega_list=omega_list,
    amplitude=amplitude,
    phase_range=phase_range,
    sequence_length=sequence_length,
    dt=dt,
    num_samples=num_samples,
    noise_std=noise_std,
    noise_value=noise_value
)

# Split into train, validation, and test sets (70/15/15 split)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)


#########################
#Visualize the dataset
##########################


# Get indices for three different classes
indices = {0: None, 1: None, 2: None}
for i in range(len(dataset)):
    sample, label = dataset[i]
    label = label.item()  # Convert tensor to scalar
    if indices[label] is None:
        indices[label] = i
    if all(v is not None for v in indices.values()):
        break

# Extract samples
samples = [dataset[idx][0] for idx in indices.values()]

print(samples[0])

times = torch.arange(samples[0].shape[0])  # Assuming time steps are along dim=0
#print(times.shape)

# Create raster plot
plt.figure(figsize=(8, 4))
for i, sample in enumerate(samples):
    #print(sample.shape)
    spike_times = times[sample.squeeze(-1).bool()]   # Get time indices where there's a spike
    plt.scatter(spike_times, [i] * len(spike_times), marker='|', s=100, label=f'Class {i}')

plt.xlabel("Time")
plt.ylabel("Sample Index")
plt.yticks([0, 1, 2], [f"Class {i}" for i in range(3)])
plt.title("Raster Plot of Delta Pulse Samples")
plt.legend()

# Save plot
save_path = os.path.join(os.getcwd(), "raster_plot.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Plot saved to {save_path}")

##########################################


# Model setup
hidden_size = 3 
num_classes = len(omega_list)
input_size = 1  # Single feature per time step

model = snn.models.SimpleResRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    adaptive_omega_a=min(omega_list) - 1.0,
    adaptive_omega_b=max(omega_list) + 1.0,
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

# Experiment setup with Weights & Biases
wandb.init(

    project="pulses-classification",
    config={
        "learning_rate": 1.0,
        "epochs": 200,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "sequence_length": sequence_length,
        "dt": dt,
        "num_samples": num_samples,
        "noise_std": noise_std,
        "omega_list": omega_list,
    },
    name=f"pulses-classification-{datetime.now().strftime('%m-%d_%H-%M-%S')}",
)

config = wandb.config

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / config.epochs)

# Training and evaluation functions
def evaluate(loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            
            print(inputs.shape)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs.permute(1, 0, 2)) 
            loss = criterion(outputs.mean(dim=0), targets)
            total_loss += loss.item()
            correct += (outputs.mean(dim=0).argmax(dim=1) == targets).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# Training loop
epochs = config.epochs
best_val_loss = float('inf')  # Initialize best validation loss for model checkpointing

for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0, 0
    # Train the model
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(inputs.permute(1, 0, 2))# Permute as expected by the model
        
        print(outputs)
       
        loss = criterion(outputs.mean(dim=0), targets)
        loss.backward()
        optimizer.step()

        print(model.out.linear.weight.data)

        train_loss += loss.item()
        train_correct += (outputs.mean(dim=0).argmax(dim=1) == targets).sum().item()
    
    # Average training loss and accuracy
    train_loss /= len(train_loader)
    train_accuracy = train_correct / len(train_loader.dataset)
    
    # Validation
    val_loss, val_accuracy = evaluate(val_loader)
    
    # Log metrics to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Train Accuracy": train_accuracy,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_accuracy,
        "Learning Rate": scheduler.get_last_lr()[0],
    })

    # Log learned omegas, b_offsets, and tau_mem
    learned_omegas = model.hidden.omega.detach().cpu().numpy()
    for i, omega in enumerate(learned_omegas):
        wandb.log({f"Neuron {i} Omega": omega, "Epoch": epoch + 1})

    learned_b = model.hidden.b_offset.detach().cpu().numpy()
    for i, b_offset in enumerate(learned_b):
        wandb.log({f"Neuron {i} b": b_offset, "Epoch": epoch + 1})

    learned_taus = model.out.tau_mem.detach().cpu().numpy()
    for i, tau in enumerate(learned_taus):
        wandb.log({f"Neuron {i} tau": tau, "Epoch": epoch + 1})

    # Print metrics
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # You can save the model state dict here if needed:
        # torch.save(model.state_dict(), 'best_model.pth')

    scheduler.step()

print("Training complete.")
wandb.finish()