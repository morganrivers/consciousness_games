import random
from time import sleep
# Initialize the bin probabilities (priors)
bin_sums = [1.0 / 11] * 11  # Uniform prior over p = 0.0, 0.1, ..., 1.0
num_iterations = 1000  # Number of observations

def ascii_plot(bin_sums, max_height=20):
    max_value = max(bin_sums)
    if max_value == 0:
        max_value = 1
    heights = [int((value / max_value) * max_height) for value in bin_sums]
    for i, height in enumerate(heights):
        p_value = i / 10  # Convert bin index to p value
        print(f"p = {p_value:3.1f}: " + '|' * height)

for iteration in range(num_iterations):
    if iteration < 300:
        rand_choice = 0
    else:
        rand_choice = random.randint(0, 1)  # Observed data point (0 or 1)

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
    sleep(0.02)  # Short delay to see the updates
    ascii_plot(bin_sums)
    print(f"\ni: {iteration}\n")
# Visualize the final distribution
ascii_plot(bin_sums)
print("Final bin probabilities:", bin_sums)

