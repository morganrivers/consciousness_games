import random
import math
from time import sleep

# Initialize the bin sums
bin_sums = [0.0] * 11
num_iterations = 1000  # number of loops

# Initialize the peak position
peak_position = 5.0  # Start from the middle
alpha = 0.05         # Smoothing factor for peak position update
sigma = 2            # Standard deviation for the Gaussian distribution

def ascii_plot(bin_sums, max_height=20):
    # Find the maximum bin sum to scale the heights
    max_value = max(bin_sums)
    if max_value == 0:
        max_value = 1
    # Scale each bin sum to the desired max_height
    heights = [int((value / max_value) * max_height) for value in bin_sums]

    # Print each bin as a vertical line of '|' characters
    for i, height in enumerate(heights):
        print(f"Bin {i:2}: " + '|' * height)

for iteration in range(num_iterations):
    if iteration < 300:
        rand_choice = 0
    else:
        rand_choice = random.randint(0, 1)  # Observed data point (0 or 1)

    # Update peak_position towards 0 or 10
    target = 0 if rand_choice == 0 else 10
    peak_position = (1 - alpha) * peak_position + alpha * target

    # Generate the Gaussian distribution centered at peak_position
    distribution = [math.exp(- ((i - peak_position) ** 2) / (2 * sigma ** 2)) for i in range(11)]

    # Normalize the distribution to ensure the sum is 1
    total_sum = sum(distribution)
    distribution = [x / total_sum for x in distribution]

    # Add the distribution to bin sums
    bin_sums = [bin_sums[i] + distribution[i] for i in range(11)]

    # Print final bin sums and ASCII plot
    ascii_plot(bin_sums)
    print(f"\ni: {iteration}\n")

    sleep(0.02)
print("Bin sums:", bin_sums)
print("Final peak position:", peak_position)