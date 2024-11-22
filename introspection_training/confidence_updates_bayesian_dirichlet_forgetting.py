import math
import random
from time import sleep

# Initialize the bin probabilities (uniform prior)
bin_sums = [1.0 / 11] * 11  # Uniform prior over p = 0.0, 0.1, ..., 1.0
num_iterations = 1000  # Number of observations

# Dirichlet smoothing factor to apply after each update (can be toggled on/off)
alpha_smoothing = 0.0000001#0.1  # Smoothing factor to prevent overconfidence
apply_dirichlet_smoothing = True  # Set to True to apply Dirichlet smoothing after each update
forgetting_factor = 1
def ascii_plot(bin_sums, max_height=20):
    max_value = max(bin_sums)
    if max_value == 0:
        max_value = 1
    heights = [int((value / max_value) * max_height) for value in bin_sums]
    for i, height in enumerate(heights):
        p_value = i / 10  # Convert bin index to p value
        print(f"p = {p_value:3.1f}: " + '|' * height)

def get_gaussian(x, mu=5, sigma=2):
    """Generates a Gaussian centered at `mu` with a standard deviation `sigma`."""
    return math.exp(-0.5 * ((x - mu) ** 2) / sigma ** 2)

gaussian = [alpha_smoothing*get_gaussian(level) for level in range(11)]
print(gaussian)
# quit()
for iteration in range(num_iterations):
    if iteration%40 < 20:
        rand_choice = 0
    else:
        rand_choice = 1#random.randint(0, 1)  # Observed data point (0 or 1)

    # Update the posterior distribution
    bin_probabilities = []
    for i, prior_p in enumerate(bin_sums):
        p_i = i / 10  # Possible p value (0.0 to 1.0 in steps of 0.1)
        # Likelihood of observing rand_choice given p_i
        likelihood = p_i if rand_choice == 1 else (1 - p_i)
        posterior_p = prior_p * likelihood
        bin_probabilities.append(posterior_p)

    # Normalize the posterior
    total = sum(bin_probabilities)
    bin_sums = [p / total for p in bin_probabilities]
    # Apply forgetting factor to smooth out influence of past observations
    bin_sums = [p * forgetting_factor for p in bin_sums]

    # Apply Dirichlet smoothing if the flag is set to True
    if apply_dirichlet_smoothing:
        bin_sums =  [x + y for x, y in zip(bin_sums, gaussian)] #[p + alpha_smoothing for p in bin_sums]
        # Normalize again after applying Dirichlet smoothing
        total = sum(bin_sums)
        bin_sums = [p / total for p in bin_sums]

    sleep(0.3)  # Short delay to see the updates
    ascii_plot(bin_sums)
    print(f"\ni: {iteration}\n")

# Visualize the final distribution
ascii_plot(bin_sums)
print("Final bin probabilities:", bin_sums)
