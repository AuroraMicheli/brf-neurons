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

os.environ['WANDB_DIR'] = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/brf-neurons/experiments/frequencies/wandb/tmp'

seed=45 #with seed 42, acc=100 after 24 epochs. with seed 50, acc=30 after >40 epochs
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
        #print(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

################################################################
# Data preparation
################################################################

rand_num = random.randint(1, 10000)
omega_list = [10.0, 17.0, 25.0]
#omega_list = [3.0, 17.0, 48.0]
omega_list_neurons = [8.0, 15.0, 28.0]
#omega_list_neurons = [5.0, 20.0, 29.0] #using this to check whether it's really sensitive to right frequencies 
#omega_list_neurons = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28]
amplitude = 1.0
phase_range = (0, 0)
#phase_range = (0, 0)
sequence_length = 250
dt = 0.01
num_samples = 3000
noise_std = 0
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

# Set up time axis
t = torch.arange(0, sequence_length * dt, dt)

# Find one sample from each class
samples = []
labels = sine_dataset.labels
data = sine_dataset.data

for class_idx in range(len(omega_list)):
    for i in range(len(labels)):
        if labels[i] == class_idx:
            samples.append(data[i].squeeze().numpy())
            break

# Plotting
plt.figure(figsize=(10, 6))
offset = 2.5  # Vertical offset between waveforms

for i, sample in enumerate(samples):
    shifted_sample = sample + i * offset  # Shift each waveform vertically
    plt.plot(t, shifted_sample, label=f'Class {i} (ω={omega_list[i]})')

# Format y-axis to show only class indices at the right locations
yticks = [i * offset for i in range(len(samples))]
ytick_labels = [f'Class {i}' for i in range(len(samples))]
plt.yticks(yticks, ytick_labels)

plt.xlabel('Time (s)')
plt.title('Sample Sine Waves from Each Class')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('sine_wave_samples.png', dpi=300)
plt.show()

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
    adaptive_omega_a=min(omega_list)- 1.0,
    adaptive_omega_b=max(omega_list)+1.0,
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

#print(model.out.linear.weight.data)
#print(model.hidden.omega.detach().cpu().numpy())
################################################################
# Experiment setup with Weights & Biases
################################################################

wandb.init(
    project="sine-wave-debug-exp",
    config={
        "learning_rate": 1.0,
        "epochs": 150,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "sequence_length": sequence_length,
        "dt": dt,
        "num_samples": num_samples,
        "noise_std": noise_std,
        "omega_list": omega_list,
        "l1_lambda": 0.001, # Regularization term for input weights
        "use_wandb": True
    },
    name=f"learn-everything-{datetime.now().strftime('%m-%d_%H-%M-%S')}",
)

config = wandb.config

criterion_1 = nn.CrossEntropyLoss()
criterion_2 = nn.MSELoss()
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
    total_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, ((hidden_z, hidden_u), out_u), num_spikes = model(inputs.permute(1, 0, 2)) 

            #print(outputs) # Adjust for [seq_len, batch, input_size]
            loss = criterion_1(outputs.mean(dim=0), targets)
            total_loss += loss.item()
            correct += (outputs.mean(dim=0).argmax(dim=1) == targets).sum().item()
            #print(f'---------\n{outputs.mean(dim=0).argmax(dim=1)}')
            #print(f'{targets}\n---------')

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

################################################################
# Training loop with wandb logging
################################################################

#reg_lambda = config.reg_lambda
epochs = config.epochs
best_val_loss = float("inf")

omega_tracking = [model.hidden.omega.detach().cpu().numpy().copy()]

for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        #print(targets)
        #print(inputs.shape)
        optimizer.zero_grad()
        outputs, ((hidden_z, hidden_u), out_u), num_spikes = model(inputs.permute(1, 0, 2)) 

        #print(outputs.shape)
        #print(outputs)
        #print(outputs.mean(dim=(0, 1)))


        loss = criterion_1(outputs.mean(dim=0), targets) #+ criterion_2(hidden_u.sum(dim=2).unsqueeze(-1).permute(1, 0, 2), inputs)
        
        '''''
        l1_lambda = config.l1_lambda
        rf_weights = model.hidden.linear.weight
        l1_reg = torch.norm(rf_weights, p=1)

        loss += l1_lambda * l1_reg
        '''''
        # Add regularization: -sum(exp(-omega_i))
        #reg_term = -torch.sum(torch.exp(-model.hidden.omega))
        #loss += reg_lambda * reg_term

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
                
        #print(model.out.linear.weight.data)
        
        total_loss += loss.item()
        correct += (outputs.mean(dim=0).argmax(dim=1) == targets).sum().item()


    #print(model.hidden.linear.weight.shape)
    #print(model.hidden.linear.weight)
    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / len(train_loader.dataset)

    val_loss, val_accuracy = evaluate(val_loader)

    #num_zero_weights = torch.sum(model.hidden.linear.weight.abs() < 1e-3).item()
    #total_weights = model.hidden.linear.weight.numel()
    #sparsity = num_zero_weights / total_weights
    #print(f"Epoch {epoch}: RFCell weight sparsity = {sparsity:.2%}")

    # Log metrics to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Train Accuracy": train_accuracy,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_accuracy,
        "Learning Rate": scheduler.get_last_lr()[0],
    })

    current_omegas = model.hidden.omega.detach().cpu().numpy().copy()
    omega_tracking.append(current_omegas)

    # Log learned omegas at each epoch
    #learned_omegas = model.neuron_layer.omegas.detach().cpu().numpy()  # Adjust based on your model's omega parameter location
    learned_omegas = model.hidden.omega.detach().cpu().numpy()
    #print(learned_omegas.shape)
    
    #wandb.log({"learned_omegas": learned_omegas, "Epoch": epoch + 1})
    #wandb.log({f"learned_omega_{i}": omega for i, omega in enumerate(learned_omegas)})


 
    for i, omega in enumerate(learned_omegas):
        wandb.log({f"Neuron {i} Omega": omega, "Epoch": epoch + 1})


    learned_b = model.hidden.b_offset.detach().cpu().numpy()
    for i, b_offset in enumerate(learned_b):
        wandb.log({f"Neuron {i} b": b_offset, "Epoch": epoch + 1})

    learned_taus = model.out.tau_mem.detach().cpu().numpy()
    for i, tau in enumerate(learned_taus):
        wandb.log({f"Neuron {i} tau": tau, "Epoch": epoch + 1})

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f},, LR: {scheduler.get_last_lr()[0]:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        #torch.save(model.state_dict(), save_path)

    scheduler.step()

print("Training complete. Best model saved to", save_path)




omega_tracking = numpy.array(omega_tracking)

# Plotting
plt.figure(figsize=(8, 5))
for i in range(omega_tracking.shape[1]):
    plt.plot(omega_tracking[:, i], label=f'Neuron {i}')

plt.xlabel('Epoch')
plt.ylabel('Omega Value')
plt.title('Omega Evolution During Training')
plt.legend()
plt.grid(True)

# Save the plot
omega_plot_path = os.path.join(os.getcwd(), "omega_waves_evol_wrong.png")
plt.savefig(omega_plot_path, dpi=300)
plt.show()

# Finalize wandb logging
wandb.finish()
