import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
import math
import random
import sys
import numpy
import wandb
import os
sys.path.append("../..")
import snn
import tools
import matplotlib.pyplot as plt

################################################################
# General settings
################################################################
seed=45 #with seed 42, acc=100 after 24 epochs. with seed 50, acc=30 after >40 epochs
random.seed(seed)
torch.manual_seed(seed)
numpy.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = device.type == "cuda"
num_workers = 1 if device.type == "cuda" else 0
print(f"Using device: {device}")

################################################################
# Custom dataset for sine wave reconstruction
################################################################

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
        self._generate_dataset()

    def _generate_dataset(self):
        t = torch.arange(0, self.sequence_length * self.dt, self.dt)
        for omega in self.omega_list:
            for _ in range(self.num_samples // len(self.omega_list)):
                phase = random.uniform(*self.phase_range)
                sine_wave = self.amplitude * torch.sin(omega * t + phase)
                noise = torch.normal(0, self.noise_std, size=sine_wave.size())
                noisy_wave = sine_wave + noise
                self.data.append(noisy_wave.unsqueeze(-1))  # Add feature dimension

        self.data = torch.stack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Input is the same as the target for reconstruction

################################################################
# Data preparation
################################################################

rand_num = random.randint(1, 10000)
omega_list = [5, 17, 25]
omega_list_neurons = [5,17,25]

amplitude = 1.0
#phase_range = (0, 2 * math.pi)
phase_range = (0, 0)
sequence_length = 200
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

################################################################
# Reconstruction model setup
################################################################

hidden_size = 3
input_size = 1  # Single feature per time step

# Define your SNN model for reconstruction
model = snn.models.SimpleResRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=input_size,  # Output size matches input size for reconstruction
    adaptive_omega_a=min(omega_list) - 0.5,
    adaptive_omega_b=max(omega_list) + 0.5,
    adaptive_b_offset_a=0.1,
    adaptive_b_offset_b=1.0,
    out_adaptive_tau_mem_mean=20.0,
    out_adaptive_tau_mem_std=1.0,
    label_last=False,  # Output the full sequence
    mask_prob=0.0,
    output_bias=True,
    initial_omegas=omega_list_neurons
).to(device)

model = torch.jit.script(model)
################################################################
# Experiment setup with Weights & Biases
################################################################

wandb.init(
    project="sine-wave-reconstruction",
    config={
        "learning_rate": 1.0,
        "epochs": 50,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "sequence_length": sequence_length,
        "dt": dt,
        "num_samples": num_samples,
        "noise_std": noise_std,
        "omega_list": omega_list,
    },
    name=f"sine-wave-reconstruction-{datetime.now().strftime('%m-%d_%H-%M-%S')}",
)

config = wandb.config

criterion = nn.MSELoss()  # Use mean squared error for reconstruction
#optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / config.epochs)

os.makedirs("models", exist_ok=True)
save_path = f"models/{datetime.now().strftime('%m-%d_%H-%M-%S')}-best-model.pt"

################################################################
# Training and evaluation functions
################################################################

def evaluate(loader):
    model.eval()
    total_loss = 0
    examples = []
    membranes=[]

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, ((hidden_z, hidden_u), out_u), num_spikes  = model(inputs.permute(1, 0, 2))  # Adjust for [seq_len, batch, input_size]
            
            #print(targets.shape)
            #print("inputs shape:", inputs.permute(1, 0, 2).shape)  #[200, 256, 1]
            #print("hidden_u shape", hidden_u.sum(dim=2).unsqueeze(-1).permute(1, 0, 2).shape)   #[200, 256, 1]
            #print(hidden_u[10].shape)
            #print(hidden_u[10])
            #print(hidden_u[:,5,:].shape)

            loss = criterion(hidden_u.sum(dim=2).unsqueeze(-1).permute(1, 0, 2), targets)
            total_loss += loss.item()

            inputs=inputs.permute(1, 0, 2)
            # Log a few examples
            for i in range(min(inputs.shape[0], 5 - len(examples))):  # Add up to 5 unique samples total
                examples.append((inputs[:, i, :].cpu(), hidden_u.sum(dim=2).unsqueeze(-1)[:, i, :].cpu()))
                membranes.append(hidden_u[:, i, :].cpu())
                if len(examples) >= 5:
                    break
            '''''
            if len(examples) < 5:  # Log up to 3 examples
                examples.append((inputs[:,5,:].cpu(), hidden_u.sum(dim=2).unsqueeze(-1)[:,5,:].cpu()))
                membranes.append(hidden_u[:,5,:].cpu())
            '''''
    avg_loss = total_loss / len(loader)
    return avg_loss, examples, membranes

################################################################
# Training loop with wandb logging
################################################################

epochs = config.epochs
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, ((hidden_z, hidden_u), out_u), num_spikes  = model(inputs.permute(1, 0, 2))
        loss = criterion(hidden_u.sum(dim=2).unsqueeze(-1).permute(1, 0, 2), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    val_loss, val_examples, membranes = evaluate(val_loader)

    # Log metrics and examples to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Validation Loss": val_loss,
    })
    #for example in val_examples:
        #visualize_reconstruction(example[0], example[1], epoch)
    
    learned_omegas = model.hidden.omega.detach().cpu().numpy()
    for i, omega in enumerate(learned_omegas):
        wandb.log({f"Neuron {i} Omega": omega, "Epoch": epoch + 1})

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        #torch.save(model.state_dict(), save_path)

    scheduler.step()

print("Training complete. Best model saved to", save_path)

################################################################
# Testing on unseen data and visualization
################################################################

test_loss, test_examples, membranes  = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}")

for i, (inputs, outputs) in enumerate(test_examples):
    plt.figure(figsize=(12, 6))
    plt.plot(inputs.numpy(), label="Ground Truth", color="blue", alpha=0.6)
    plt.plot(outputs.numpy(), label="Reconstruction", color="red", alpha=0.8)
    membrane = membranes[i]  # shape: [seq_len, num_neurons] (e.g. [200, 3])
    for j in range(membrane.shape[1]):  # iterate over neurons
        plt.plot(membrane[:, j].numpy(), label=f"Membrane Neuron {j}", alpha=0.5)

    plt.legend()
    plt.title(f"Test Example {i+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"reconstruction_example_{i+1}.png")
    plt.close()
