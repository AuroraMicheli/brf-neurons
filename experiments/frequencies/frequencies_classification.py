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
print(matplotlib.__version__)
import wandb
import os
sys.path.append("../..")
import snn
import tools

seed=50 #with seed 42, acc=100 after 24 epochs. with seed 50, acc=30 after >40 epochs
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
# Custom dataset for sine wave classification
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
omega_list = [10.0, 27.0, 46.0]
amplitude = 1.0
phase_range = (0, 2 * math.pi)
sequence_length = 100
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
# Model setup
################################################################

hidden_size = 3 
num_classes = len(omega_list)
input_size = 1  # Single feature per time step

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
    initial_omegas=omega_list
).to(device)

model = torch.jit.script(model)

################################################################
# Experiment setup with Weights & Biases
################################################################

wandb.init(
    project="sine-wave-adhoc-omega-initialization",
    config={
        "learning_rate": 0.1,
        "epochs": 300,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "sequence_length": sequence_length,
        "dt": dt,
        "num_samples": num_samples,
        "noise_std": noise_std,
        "omega_list": omega_list,
    },
    name=f"sine-wave-adhoc-initialization-{datetime.now().strftime('%m-%d_%H-%M-%S')}",
)

config = wandb.config

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / config.epochs)


os.makedirs("models", exist_ok=True)

save_path = f"models/{datetime.now().strftime('%m-%d_%H-%M-%S')}-best-model.pt"

################################################################
# Training and evaluation functions
################################################################

def evaluate(loader):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs.permute(1, 0, 2)) 
            #print(outputs.shape) # Adjust for [seq_len, batch, input_size]
            loss = criterion(outputs.mean(dim=0), targets)
            total_loss += loss.item()
            correct += (outputs.mean(dim=0).argmax(dim=1) == targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

################################################################
# Training loop with wandb logging
################################################################

epochs = config.epochs
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(inputs.permute(1, 0, 2))
        loss = criterion(outputs.mean(dim=0), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.mean(dim=0).argmax(dim=1) == targets).sum().item()

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / len(train_loader.dataset)

    val_loss, val_accuracy = evaluate(val_loader)

    # Log metrics to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Train Accuracy": train_accuracy,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_accuracy,
    })

    # Log learned omegas at each epoch
    #learned_omegas = model.neuron_layer.omegas.detach().cpu().numpy()  # Adjust based on your model's omega parameter location
    learned_omegas = model.hidden.omega.detach().cpu().numpy()
    for i, omega in enumerate(learned_omegas):
        wandb.log({f"Neuron {i} Omega": omega, "Epoch": epoch + 1})

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)

    scheduler.step()

print("Training complete. Best model saved to", save_path)

# Finalize wandb logging
wandb.finish()
