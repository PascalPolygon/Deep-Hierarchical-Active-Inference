import re
import numpy as np
import matplotlib.pyplot as plt

# Function to parse losses from the file
def parse_losses(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    episode_pattern = re.compile(r'=== Episode \d+ ===')
    high_level_pattern = re.compile(r'> High-Level Train epoch (\d+) \[ensemble ([\d\.-]+) \| reward ([\d\.-]+)\]')

    ensemble_losses = []
    reward_losses = []
    current_episode_ensemble = []
    current_episode_reward = []

    for line in lines:
        episode_match = episode_pattern.match(line)
        high_level_match = high_level_pattern.match(line)

        if episode_match:
            if current_episode_ensemble and current_episode_reward:
                ensemble_losses.append(current_episode_ensemble)
                reward_losses.append(current_episode_reward)
                current_episode_ensemble = []
                current_episode_reward = []

        if high_level_match:
            epoch = int(high_level_match.group(1))
            ensemble_loss = float(high_level_match.group(2))
            reward_loss = float(high_level_match.group(3))

            current_episode_ensemble.append(ensemble_loss)
            current_episode_reward.append(reward_loss)

    # Append the last episode
    if current_episode_ensemble and current_episode_reward:
        ensemble_losses.append(current_episode_ensemble)
        reward_losses.append(current_episode_reward)

    # Debug: print parsed data
    print(f"Parsed from {file_path}:")
    print(f"Ensemble Losses: {ensemble_losses}")
    print(f"Reward Losses: {reward_losses}")
    print()

    return np.array(ensemble_losses), np.array(reward_losses)

# Function to plot losses with IQR
def plot_losses(dhai_losses, flat_losses, title, ylabel):
    epochs = np.arange(20, 201, 20)

    # Check if the losses arrays are empty
    if dhai_losses.size == 0 or flat_losses.size == 0:
        print(f"Error: One of the loss arrays is empty. DHAI losses size: {dhai_losses.size}, Flat losses size: {flat_losses.size}")
        return

    dhai_median = np.median(dhai_losses, axis=0)
    dhai_iqr = np.percentile(dhai_losses, [25, 75], axis=0)
    
    flat_median = np.median(flat_losses, axis=0)
    flat_iqr = np.percentile(flat_losses, [25, 75], axis=0)

    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, dhai_median, label='DHAI Median', color='blue')
    plt.fill_between(epochs, dhai_iqr[0], dhai_iqr[1], color='blue', alpha=0.3, label='DHAI IQR')

    plt.plot(epochs, flat_median, label='Flat Median', color='green')
    plt.fill_between(epochs, flat_iqr[0], flat_iqr[1], color='green', alpha=0.3, label='Flat IQR')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Parsing both files
dhai_ensemble_losses, dhai_reward_losses = parse_losses('log_debug_0/out.txt')
flat_ensemble_losses, flat_reward_losses = parse_losses('log_debug_0/out_og.txt')

# Plotting Ensemble Loss
plot_losses(dhai_ensemble_losses, flat_ensemble_losses, 'Ensemble Loss Comparison', 'Loss')

# Plotting Reward Loss
plot_losses(dhai_reward_losses, flat_reward_losses, 'Reward Loss Comparison', 'Loss')
