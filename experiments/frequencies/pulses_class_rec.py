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

omega_list = [ 19.0, 26.0, 38]
#omega_lst = [16, 27, 36]
omega_list_neurons =  [17.0, 31.0, 42]


#omega_list_neurons = omega_list
amplitude = 10.0
phase_range = (0, 2*math.pi)  # Full range of phase shifts
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
save_path = os.path.join(os.getcwd(), "raster_plot_multi.png")
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
    #adaptive_b_offset_a=0.1, #original
    #adaptive_b_offset_b=1.0, #original
    adaptive_b_offset_a=2., #2
    adaptive_b_offset_b=3., #3
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
        "epochs": 20,
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

def spike_rate_loss(inputs, hidden_z, comp_weight=0.1, eps=1e-8):
  
    #inputs [256,250,1]
    #hidden_z [256,250,3]
    # --- Rate matching (per-sample) ---
    input_spike_count = inputs.sum(dim=1).squeeze(-1)  # Sum over time, for each sample (batch_size,1)
    output_spike_count = hidden_z.sum(dim=(1,2)) # Sum over time, for each sample (batch_size,neurons)
    #print(input_spike_count) #256
    #print(output_spike_count) #256

    normalized_rate_loss = ((input_spike_count - output_spike_count) / (input_spike_count + eps)) ** 2




    # --- Competition (per-sample) ---
    total_spikes_per_neuron = hidden_z.sum(dim=1) #[batch, neurons]
    max_spikes, _ = total_spikes_per_neuron.max(dim=1, keepdim=True) # [batch, 1]

    #print(total_spikes_per_neuron)
   # Compute the competition loss: penalize if more than one neuron has spikes
    comp_loss = (total_spikes_per_neuron.sum(dim=1) - max_spikes.squeeze()).mean()

    total_loss = normalized_rate_loss.mean() #+ comp_loss  # Average over the batch

    #return my_loss.mean()
    return total_loss

def gaussian_kernel(size: int, std: float):
    """Generates a 1D Gaussian kernel."""
    t = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    gauss = torch.exp(-0.5 * (t / std) ** 2)
    gauss /= gauss.sum()
    return gauss


def spike_kernel_mse(inputs, hidden_z, kernel_size=21, std=1.0):
  
    #inputs [256,250,1]
    #hidden_z [256,250,3]

    # --- Rate matching (per-sample) ---
    input_spike_count = inputs.squeeze(-1)  #  for each sample (batch_size,time steps)
    output_spike_count = hidden_z.sum(dim=2) # Sum over time, for each sample (batch_size,time steps)
    #print(input_spike_count.shape) #[256,250]
    #print(output_spike_count.shape) #[256,250]

    kernel = gaussian_kernel(kernel_size, std).to(inputs.device)
    kernel = kernel.view(1, 1, -1)  # shape [1, 1, kernel_size]

    # Reshape to [batch, channel=1, time] for conv1d
    inputs = input_spike_count.unsqueeze(1)     # [batch, 1, time]
    hidden_z =  output_spike_count.unsqueeze(1) # [batch, 1, time]
        # Apply 1D convolution (padding to keep size)
    padding = kernel_size // 2
    smooth_input = nn.functional.conv1d(inputs, kernel, padding=padding)
    smooth_output = nn.functional.conv1d(hidden_z, kernel, padding=padding)

    # Compute MSE between the smoothed spike trains (MASKED)
    squared_error = (smooth_output - smooth_input)**2
    masked_loss = squared_error * smooth_input
    loss = masked_loss.mean()
    #loss = nn.functional.mse_loss(smooth_output, smooth_input)

    return loss

def membrane_kernel_mse(inputs, hidden_u, kernel_size=21, std=1.0):
  
    #inputs [256,250,1]
    #hidden_z [256,250,3]

    # --- Rate matching (per-sample) ---
    input_spike_count = inputs.squeeze(-1)  #  for each sample (batch_size,time steps)
    output_membrane = hidden_u.sum(dim=2) # Sum over time, for each sample (batch_size,time steps)
    #print(input_spike_count.shape) #[256,250]
    #print(output_spike_count.shape) #[256,250]

    kernel = gaussian_kernel(kernel_size, std).to(inputs.device)
    kernel = kernel.view(1, 1, -1)  # shape [1, 1, kernel_size]

    # Reshape to [batch, channel=1, time] for conv1d
    inputs = input_spike_count.unsqueeze(1)     # [batch, 1, time]
    hidden_u =  output_membrane.unsqueeze(1) # [batch, 1, time]
        # Apply 1D convolution (padding to keep size)
    padding = kernel_size // 2
    smooth_input = nn.functional.conv1d(inputs, kernel, padding=padding)
    #smooth_output = nn.functional.conv1d(hidden_z, kernel, padding=padding)

    # Compute MSE between the smoothed spike trains (MASKED)
    squared_error = (hidden_u - smooth_input)**2
    masked_loss = squared_error * smooth_input
    loss = masked_loss.mean()
    #loss = nn.functional.mse_loss(smooth_output, smooth_input)

    return loss

def spike_kernel_mse_competitive_masked(inputs, hidden_z, kernel_size=21, std=1.0, eps=1e-8):
    """
    Competitive spike kernel loss (masked):
    - smoothes input and output spike trains with a Gaussian kernel
    - finds the best matching neuron per sample
    - encourages winner to match input
    - suppresses others
    """

    # inputs: [batch, time, 1]
    # hidden_z: [batch, time, neurons]

    batch_size, time_steps, n_neurons = hidden_z.shape

    # --- Smooth input ---
    input_spike_count = inputs.squeeze(-1)  # [batch, time]

    kernel = gaussian_kernel(kernel_size, std).to(inputs.device)
    kernel = kernel.view(1, 1, -1)  # [1, 1, kernel_size]
    padding = kernel_size // 2

    smooth_input = nn.functional.conv1d(
        input_spike_count.unsqueeze(1), kernel, padding=padding
    ).squeeze(1)  # [batch, time]

    input_mask = smooth_input.detach()  # [batch, time]

    # --- Smooth each neuron's spike train ---
    hidden_z = hidden_z.permute(0, 2, 1)  # [batch, neurons, time]
    kernel = kernel.repeat(n_neurons, 1, 1)  # [n_neurons, 1, kernel_size]
    smooth_output = nn.functional.conv1d(
        hidden_z, kernel, padding=padding, groups=n_neurons
    )  # [batch, neurons, time]

    # --- Compute masked MSE per neuron ---
    squared_error = (smooth_output - smooth_input.unsqueeze(1)) ** 2  # [batch, neurons, time]
    masked_error = squared_error * input_mask.unsqueeze(1)            # [batch, neurons, time]
    #mse_per_neuron = masked_error.mean(dim=2)                         # [batch, neurons]
    mse_per_neuron = masked_error.sum(dim=2) #sum over time instead of mean

    # --- Winner-takes-all ---
    winner_idx = mse_per_neuron.argmin(dim=1)  # [batch]
    winner_mask = torch.nn.functional.one_hot(winner_idx, num_classes=n_neurons).float()  # [batch, neurons]

    # --- Encourage winner to match ---
    winner_outputs = (smooth_output * winner_mask.unsqueeze(-1)).sum(dim=1)  # [batch, time]
    #print(winner_outputs)
    match_error = ((winner_outputs - smooth_input) ** 2) * input_mask
    #match_loss = match_error.mean()
    match_loss = match_error.sum(dim=1).mean(dim=0)

    # --- Suppress non-winners ---
    non_winner_mask = 1.0 - winner_mask  # [batch, neurons]
    suppressed_outputs = smooth_output * non_winner_mask.unsqueeze(-1)  # [batch, neurons, time]
    #suppress_loss = (suppressed_outputs ** 2 * input_mask.unsqueeze(1)).mean()
    suppress_loss = (suppressed_outputs ** 2 * input_mask.unsqueeze(1)).sum(dim=(1,2)).mean() #sum over time and neurons instead of mean
    
    total_loss = match_loss + suppress_loss
    #print(match_loss)
    #print(suppress_loss)
    return total_loss


def membrane_kernel_mse_competitive(inputs, hidden_u, kernel_size=21, std=1.0, eps=1e-8):
    """
    Competitive membrane potential loss:
    - encourages the best matching neuron to fit the input trace
    - suppresses others to have low output
    """

    # inputs: [batch, time, 1]
    # hidden_u: [batch, time, neurons]

    batch_size, time_steps, n_neurons = hidden_u.shape

    input_spike_count = inputs.squeeze(-1)  # [batch, time]

    kernel = gaussian_kernel(kernel_size, std).to(inputs.device)
    kernel = kernel.view(1, 1, -1)

    # Smooth input with Gaussian
    smooth_input = nn.functional.conv1d(input_spike_count.unsqueeze(1), kernel, padding=kernel_size // 2)  # [batch, 1, time]
    smooth_input = smooth_input.squeeze(1)  # [batch, time]

    # Transpose for easier broadcasting: hidden_u -> [batch, neurons, time]
    hidden_u = hidden_u.permute(0, 2, 1)  # [batch, neurons, time]

    input_mask = smooth_input.detach()

    # Compute MSE for each neuron's output vs input
    squared_error = (hidden_u - smooth_input.unsqueeze(1)) ** 2  # [batch, neurons, time]
    masked_error = squared_error * input_mask.unsqueeze(1)       # [batch, neurons, time]
    mse_per_neuron = masked_error.mean(dim=2)                    # [batch, neurons]

    # Identify best matching neuron (minimum error)
    winner_idx = mse_per_neuron.argmin(dim=1)  # [batch]

    # Construct winner mask: [batch, neurons]
    winner_mask = torch.nn.functional.one_hot(winner_idx, num_classes=n_neurons).float()

    # Match loss: MSE for winning neuron
    # Winner output: [batch, time]
    winner_outputs = (hidden_u * winner_mask.unsqueeze(-1)).sum(dim=1)
    match_error = ((winner_outputs - smooth_input) ** 2) * input_mask
    #match_loss = match_error.mean()
    match_loss = match_error.sum(dim=1).mean(dim=0)

    # Suppress others
    non_winner_mask = 1.0 - winner_mask  # [batch, neurons]
    suppressed_outputs = hidden_u * non_winner_mask.unsqueeze(-1)  # [batch, neurons, time]
    #suppress_loss = (suppressed_outputs ** 2 * input_mask.unsqueeze(1)).mean()
    suppress_loss = (suppressed_outputs ** 2 * input_mask.unsqueeze(1)).sum(dim=(1,2)).mean()
    # Combine losses
    total_loss = match_loss + suppress_loss
    #print(match_loss)
    #print(suppress_loss)
    return total_loss


criterion_1 = nn.CrossEntropyLoss()
#criterion_2 = soft_peak_valley_loss
#optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

optimizer = torch.optim.SGD([
    {'params': [model.hidden.omega], 'lr': config.learning_rate * 10},  # 10x for omega
    {'params': [p for n, p in model.named_parameters() if n != 'hidden.omega'], 'lr': config.learning_rate}
], momentum=0.9)

'''
optimizer = torch.optim.Adam([
    {'params': [model.hidden.omega], 'lr': config.learning_rate * 5},  # boosted LR for omega
    {'params': [p for n, p in model.named_parameters() if n != 'hidden.omega'], 'lr': config.learning_rate}
])
'''
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

omega_tracking = [model.hidden.omega.detach().cpu().numpy().copy()]

for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0, 0
    # Train the model
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, ((hidden_z, hidden_u), out_u), num_spikes = model(inputs.permute(1, 0, 2))# Permute as expected by the model
        

        #loss = criterion_1(outputs.mean(dim=0), targets) + criterion_2(inputs,hidden_u.sum(dim=2).unsqueeze(-1).permute(1, 0, 2))

        #SPIKE RATE-COMPETITION
        #print(hidden_z.permute(1, 0, 2).shape)
        spike_loss = spike_rate_loss(inputs/amplitude, hidden_z.permute(1, 0, 2))


        smooth_kernel_loss = spike_kernel_mse(inputs/amplitude,  hidden_z.permute(1, 0, 2))
        compt_smooth_kernel_loss = spike_kernel_mse_competitive_masked(inputs/amplitude, hidden_u.permute(1, 0, 2) )

        membrane_kernel_mse_value = membrane_kernel_mse(inputs/amplitude, hidden_u.permute(1, 0, 2) )
        comp_membrane_kernel_mse_value = membrane_kernel_mse_competitive(inputs/amplitude, hidden_u.permute(1, 0, 2) )

        #loss=smooth_kernel_loss + total_loss
        #print("kernel spikes:", smooth_kernel_loss)
        #print("kernel potentials:", membrane_kernel_mse_value)
        #print("spike count:", spike_loss)
        print("class:", criterion_1(outputs.mean(dim=0), targets))
        print("reconstr:",comp_membrane_kernel_mse_value )
        loss = criterion_1(outputs.mean(dim=0), targets) + comp_membrane_kernel_mse_value #+ membrane_kernel_mse_value + 0.1*spike_loss #+ smooth_kernel_loss
        print(loss)
        #print(loss)



        loss.backward()
        '''
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: grad norm = {param.grad.norm() if param.grad is not None else 'None'}")

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm().item():.4e}")

        '''
        optimizer.step()

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

    current_omegas = model.hidden.omega.detach().cpu().numpy().copy()
    omega_tracking.append(current_omegas)


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
    plt.savefig(f"multi-pulse_reconstruction_example_{i+1}.png")
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
    plt.savefig(f"multi-raster_plot_example_{i+1}.png")
    plt.close()


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
omega_plot_path = os.path.join(os.getcwd(), "omega_pulses_evolution_right.png")
plt.savefig(omega_plot_path, dpi=300)
plt.show()
'''


####################VISUALIZE KERNELS################################

def convolve_spikes_with_kernel(spikes, kernel):
    """Convolves spike trains with a kernel along the time dimension."""
    # spikes: [T, N] → [1, N, T]
    spikes = spikes.T.unsqueeze(0)
    kernel = kernel.view(1, 1, -1)
    convolved = nn.functional.conv1d(spikes, kernel, padding=kernel.shape[-1] // 2, groups=1)
    return convolved.squeeze(0).T  # Return shape [T, N]

sample, label = dataset[0]
sample = sample.to(device).unsqueeze(0)  # shape [1, T, 1]

# Run model
model.eval()
with torch.no_grad():
    outputs, ((hidden_z, hidden_u), out_u), num_spikes = model(sample)  # model returns output spikes [1, T, N]
    #output = out['z'] if isinstance(out, dict) else out  # if model returns dict
    output = hidden_z.squeeze(0)  # shape [T, N]

#print(output.sum(dim=(0,1))) #{250,1]}
# Get input spike train
input_spikes = sample.squeeze(0).squeeze(-1)  # shape [T]

#print(input_spikes.shape) #[250]

# Generate kernel and convolve
kernel = gaussian_kernel(size=21, std=1.0).to(device)
input_conv = convolve_spikes_with_kernel(input_spikes.unsqueeze(1), kernel)
output_conv = convolve_spikes_with_kernel(output, kernel)

# Time axis
time = torch.arange(input_spikes.shape[0]) * dt

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Top: input spikes + kernel
axs[0].imshow(input_spikes.unsqueeze(1).cpu().T, aspect='auto', cmap='Greys', interpolation='nearest', extent=[0, sequence_length * dt, 0, 1])
axs[0].plot(time.cpu(), input_conv.cpu(), color='red', label='Input convolved')
axs[0].set_ylabel('Input Neuron')
axs[0].legend()

# Bottom: output spikes + kernel
axs[1].imshow(output.cpu().T, aspect='auto', cmap='Greys', interpolation='nearest', extent=[0, sequence_length * dt, 0, 1])
axs[1].plot(time.cpu(), output_conv.sum(dim=1).cpu(), color='blue', label='Output convolved (sum over neurons)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Output Neurons')
axs[1].legend()

plt.tight_layout()

# Save
save_path = os.path.join(os.getcwd(), "kernel.png")
plt.savefig(save_path, dpi=300)
plt.close()
'''

print("Training complete.")
wandb.finish()