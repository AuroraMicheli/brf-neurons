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
                delta_pulses = torch.zeros_like(sine_wave)
                peaks = (sine_wave[1:-1] > sine_wave[:-2]) & (sine_wave[1:-1] > sine_wave[2:])
                delta_pulses[1:-1][peaks] = amplitude  # Set delta pulses at peaks
                
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
omega_list = [ 27.0]
#omega_lst = [16, 27, 36]
omega_list_neurons = [27.0]
#omega_list_neurons = omega_list
amplitude = 10.0
phase_range = (0, 0)  # Full range of phase shifts
sequence_length = 250
dt = 0.01
num_samples = 3000
noise_std = 0.0

noise_value = 0.0
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


# Get indices for existing classes
unique_labels = torch.unique(dataset.labels).tolist()
indices = {}
for i in range(len(dataset)):
    sample, label = dataset[i]
    label = label.item()
    if label not in indices:
        indices[label] = i
    if len(indices) == len(unique_labels):
        break

# Extract samples
samples = [dataset[idx][0] for idx in indices.values()]
times = torch.arange(samples[0].shape[0])  # Assuming time steps are along dim=0

# Create raster plot
plt.figure(figsize=(8, 4))
for i, sample in enumerate(samples):
    spike_times = times[sample.squeeze(-1).bool()]
    plt.scatter(spike_times, [i] * len(spike_times), marker='|', s=100, label=f'Class {i}')

plt.xlabel("Time")
plt.ylabel("Sample Index")
plt.yticks(range(len(samples)), [f"Class {i}" for i in range(len(samples))])
plt.title("Raster Plot of Delta Pulse Samples")
plt.legend()

# Save plot
save_path = os.path.join(os.getcwd(), "raster_plot.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Plot saved to {save_path}")

##########################################


# Model setup
hidden_size = 1
num_classes = len(omega_list)
input_size = 1  # Single feature per time step

model = snn.models.SimpleResRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    adaptive_omega_a=min(omega_list) - 1.0,
    adaptive_omega_b=max(omega_list) + 1.0,
    #adaptive_b_offset_a=0.1, #original
    #adaptive_b_offset_b=1.0, #original
    adaptive_b_offset_a=0., #2
    adaptive_b_offset_b=0., #3
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
        "epochs": 15,
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

###################
#LIST OF LOSS FUNCTIONS TRIED
###################
def hamming_loss(y_true, y_pred):
    #[256,250,1]
    y_pred_bin = (y_pred > 0).float()
    mismatches = torch.abs(y_true - y_pred_bin).sum(dim=1)  # shape: [batch_size, 1]
    return mismatches.mean()
    #return torch.sum(torch.abs(y_true - y_pred_bin))

def cosine_similarity_loss(y_true, y_pred):
    #y_pred_bin = (y_pred > 0).float()
    y_true_flat = y_true.squeeze(-1)
    y_pred_flat = y_pred.squeeze(-1)
    cos_sim = nn.functional.cosine_similarity(y_pred_flat, y_true_flat, dim=1)
    return cos_sim.mean()
'''''
def spike_rate_loss(inputs, hidden_z, comp_weight=0.1, eps=1e-8):
    """
    inputs: [time_steps, batch_size, 1]
    hidden_z: [time_steps, batch_size, n_neurons]
    """
    # --- Rate matching ---
    input_spike_count = inputs.sum()
    output_spike_count = hidden_z.sum()
    rate_loss = ((input_spike_count / (inputs.numel() + eps)) - (output_spike_count / (hidden_z.numel() + eps))) ** 2
    #print(rate_loss)
    # --- Competition (per-sample) ---
    batch_neuron_spikes = hidden_z.sum(dim=0)  # [batch, neurons]
    max_vals, _ = batch_neuron_spikes.max(dim=-1, keepdim=True)
    comp_loss = (batch_neuron_spikes.sum(dim=-1, keepdim=True) - max_vals).mean()
    #print(comp_loss)

    total_loss = rate_loss + comp_weight * comp_loss
    return total_loss
'''''
def spike_rate_loss(inputs, hidden_z, comp_weight=0.1, eps=1e-8):
  
    #inputs [256,250,1]
    #hidden_z [256,250,3]
    # --- Rate matching (per-sample) ---
    input_spike_count = inputs.sum(dim=1).squeeze(-1)  # Sum over time, for each sample (batch_size,1)
    output_spike_count = hidden_z.sum(dim=(1,2)) # Sum over time, for each sample (batch_size,neurons)
    #print(input_spike_count.shape) #256
    #print(output_spike_count.shape) #256
    
    num_time_steps = inputs.shape[1]  # Number of time steps (250 in this case)
    # Compute rate loss for each sample
    rate_loss = ((input_spike_count / (input_spike_count.numel() + eps)) - 
                (output_spike_count / (output_spike_count.numel() + eps))) ** 2

    normalized_rate_loss = ((input_spike_count - output_spike_count) ** 2)/(input_spike_count ** 2 + eps)
    
    my_loss = torch.abs((input_spike_count - output_spike_count))
    #print(my_loss.mean())



    # --- Competition (per-sample) ---
    total_spikes_per_neuron = hidden_z.sum(dim=1) #[batch, neurons]
    max_spikes, _ = total_spikes_per_neuron.max(dim=1, keepdim=True) # [batch, 1]

    #print(total_spikes_per_neuron)
   # Compute the competition loss: penalize if more than one neuron has spikes
    comp_loss = (total_spikes_per_neuron.sum(dim=1) - max_spikes.squeeze()).mean()
    #print((total_spikes_per_neuron.sum(dim=1) == max_spikes.squeeze()))
    # Total loss (rate matching + competition)
    #print(rate_loss.mean())
    #print("competition", comp_loss)
    total_loss = normalized_rate_loss.mean() #+ comp_loss  # Average over the batch

    return my_loss

def shift_invariant_mse(target, pred, max_lag=3):
    pred = (pred > 0).float()
    losses = [
        torch.mean((torch.roll(pred, shifts=lag, dims=-1) - target)**2)
        for lag in range(-max_lag, max_lag + 1)
    ]
    return torch.min(torch.stack(losses))



criterion_1 = nn.CrossEntropyLoss()
criterion_2 = nn.MSELoss()
#criterion_2 = soft_peak_valley_loss
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / config.epochs)

# Training and evaluation functions
def evaluate(loader):
    model.eval()
    total_loss, correct = 0, 0
    examples = []
    membranes=[]
    spikes=[]
    with torch.no_grad():
        for inputs, targets in loader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, ((hidden_z, hidden_u), out_u), num_spikes = model(inputs.permute(1, 0, 2)) 
            loss = criterion_1(outputs.mean(dim=0), targets)
            total_loss += loss.item()
            correct += (outputs.mean(dim=0).argmax(dim=1) == targets).sum().item()

            inputs=inputs.permute(1, 0, 2)
            for i in range(min(inputs.shape[0], 5 - len(examples))):  # Add up to 5 unique samples total
                examples.append((inputs[:, i, :].cpu(), hidden_u.sum(dim=2).unsqueeze(-1)[:, i, :].cpu()))
                membranes.append(hidden_u[:, i, :].cpu())
                spikes.append(hidden_z[:, i, :].cpu())
                if len(examples) >= 5:
                    break

    return total_loss / len(loader), correct / len(loader.dataset), examples, membranes, spikes

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
        outputs, ((hidden_z, hidden_u), out_u), num_spikes = model(inputs.permute(1, 0, 2))# Permute as expected by the model
        

        #loss = criterion_1(outputs.mean(dim=0), targets) + criterion_2(inputs,hidden_u.sum(dim=2).unsqueeze(-1).permute(1, 0, 2))
        #####RECONSTRUCTION#######
        reconstructed = hidden_u.sum(dim=2).unsqueeze(-1).permute(1, 0, 2)  # [batch, time, input]
        squared_diff = ((inputs - reconstructed) ** 2)/amplitude
        masked_loss = (squared_diff*inputs).mean()  # weighted by the input

        #print(inputs.shape) # [256, 250, 1]
        #print(hidden_z.sum(dim=2).unsqueeze(-1).permute(1, 0, 2).shape)  # [256, 250, 1]
        #HAMMING
        hamming_loss_value = hamming_loss(inputs/amplitude, hidden_z.sum(dim=2).unsqueeze(-1).permute(1, 0, 2))
        #print(hamming_loss_value)
        
        #COSINE SIMILARITY
        cosine_similarity_loss_value = cosine_similarity_loss(inputs/amplitude, hidden_z.sum(dim=2).unsqueeze(-1).permute(1, 0, 2))
        #print(cosine_similarity_loss_value )

        #SPIKE RATE-COMPETITION
        #print(hidden_z.permute(1, 0, 2).shape)
        total_loss = spike_rate_loss(inputs/amplitude, hidden_z.permute(1, 0, 2))
        #print(total_loss)

        shift_invariant_mse_value = shift_invariant_mse(inputs/amplitude,hidden_z.sum(dim=2).unsqueeze(-1).permute(1, 0, 2))
        #print(shift_invariant_mse_value)
        #print(hidden_z.sum(dim=2).unsqueeze(-1).permute(1, 0, 2).shape)  [256, 250, 1]
        #loss = total_loss.mean() #criterion_1(outputs.mean(dim=0), targets) 
        loss=shift_invariant_mse_value
        print(loss)
        #print(loss)



        #loss.backward()
        '''''      
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: grad norm = {param.grad.norm() if param.grad is not None else 'None'}")

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm().item():.4e}")

        '''''
        #optimizer.step()

        #print(model.out.linear.weight.data)

        train_loss += loss.item()
        train_correct += (outputs.mean(dim=0).argmax(dim=1) == targets).sum().item()
    
    # Average training loss and accuracy
    train_loss /= len(train_loader)
    train_accuracy = train_correct / len(train_loader.dataset)
    
    # Validation
    val_loss, val_accuracy, examples, membranes, spikes = evaluate(val_loader)
    
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

    #scheduler.step()

for i, (inputs, outputs) in enumerate(examples):
    plt.figure(figsize=(12, 6))
    plt.plot((inputs/amplitude).numpy(), label="Ground Truth", color="blue", alpha=0.6)
    plt.plot(outputs.numpy(), label="Reconstruction", color="red", alpha=0.8)
    membrane = membranes[i]  # shape: [seq_len, num_neurons] (e.g. [200, 3])
    for j in range(membrane.shape[1]):  # iterate over neurons
        plt.plot(membrane[:, j].numpy(), label=f"Membrane Neuron {j}", alpha=0.5)

    plt.legend()
    plt.title(f"Test Example {i+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"pulse_reconstruction_example_{i+1}.png")
    plt.close()



threshold = 0  # define your spike threshold

for i, (inputs, outputs) in enumerate(examples[:]):  # create only 3 figures
    plt.figure(figsize=(10, 4))
    
    raster_data = []
    labels = []

    # Input spikes (normalized and thresholded analog signal)
    input_signal = inputs[:, 0] / amplitude
    input_spike_times = (input_signal > 0.0).nonzero(as_tuple=True)[0].numpy()
    raster_data.append(input_spike_times)
    labels.append("Input")

    # Output spikes for up to 3 neurons
    spike_tensor = spikes[i]  # shape [T, n_neurons]
    num_neurons_to_plot = min(3, spike_tensor.shape[1])
    for j in range(num_neurons_to_plot):
        neuron_spike_times = (spike_tensor[:, j] > 0).nonzero(as_tuple=True)[0].numpy()
        raster_data.append(neuron_spike_times)
        labels.append(f"Neuron {j}")

    # Plot raster: input at the top (unit 0), neurons below (unit 1, 2, ...)
    for unit_idx, times in enumerate(raster_data):
        y_position = len(raster_data) - 1 - unit_idx  # flip vertically so input is at the top
        plt.scatter(times, [y_position] * len(times), s=10, label=labels[unit_idx], alpha=0.7)

    # Styling
    yticks = list(range(len(raster_data)))
    ytick_labels = list(reversed(labels))  # input at top
    plt.yticks(yticks, ytick_labels)
    plt.grid("on", linestyle="--", alpha=0.5)
    plt.xlabel("Time Step")
    plt.ylabel("Unit")
    plt.title(f"Raster Plot - Example {i+1}")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"raster_plot_example_{i+1}.png")
    plt.close()


print("Training complete.")
wandb.finish()