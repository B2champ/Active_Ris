import numpy as np
import matplotlib.pyplot as plt

# Load the numpy data
data = np.load("/home/btech/2024/bittu.mishra/project/cwd/RIS_MISO_DRL/Fwd_Active/GroupConnected1/Learning Curves/power/8_16_5_10.0_0.001_1e-05_episode_62.npy")  # Replace "your_data.npy" with the actual file path

# Calculate the mean and standard deviation across all traces (axis=0 assumes traces are along axis 0)
mean_data = np.mean(data, axis=0)
std_data = np.std(data, axis=0)

# Calculate a moving average to smooth the mean
window_size = 5
moving_avg = np.convolve(mean_data, np.ones(window_size) / window_size, mode='valid')

# Plotting
plt.figure(figsize=(10, 6))

# Plot the mean with shaded standard deviation
plt.plot(mean_data, label="Mean")
plt.fill_between(range(len(mean_data)), mean_data - std_data, mean_data + std_data, alpha=0.2, label="Std Dev")

# Plot the moving average
plt.plot(range(len(moving_avg)), moving_avg, label="Moving Average (window=5)", color='orange')

# Add labels and title
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Smoothed Learning Curve with Mean and Standard Deviation")
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig("smoothed_learning_curve.png", dpi=300)

plt.show()
