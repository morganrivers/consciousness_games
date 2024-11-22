import math
import random

# Number of categories (0 to 10)
num_categories = 11
mean_category = 3  # Center the Gaussian distribution at category 5
std_dev = 1.0  # Standard deviation for the Gaussian distribution

def ascii_plot(bin_sums, max_height=20):
    max_value = max(bin_sums)
    if max_value == 0:
        max_value = 1
    heights = [int((value / max_value) * max_height) for value in bin_sums]
    for i, height in enumerate(heights):
        p_value = i / 10  # Convert bin index to p value
        print(f"p = {p_value:3.1f}: " + '|' * height)

# Generate static probabilities using a Gaussian distribution centered around category 5
def gaussian_probabilities(mean, std_dev, num_categories):
    probabilities = [math.exp(-0.5 * ((i - mean) / std_dev) ** 2) for i in range(num_categories)]
    sum_probabilities = sum(probabilities)
    probabilities = [p / sum_probabilities for p in probabilities]
    return probabilities

# Generate static probabilities centered around category 5
def just_the_mean(mean, num_categories):
    probabilities = [0] * num_categories
    probabilities[mean] = 1
    return probabilities

# Generate static probabilities centered around category 5
def uniform(num_categories):
    probabilities = [1] * num_categories
    sum_probabilities = sum(probabilities)
    probabilities = [p / sum_probabilities for p in probabilities]
    return probabilities

# Function to compute the expected prediction
def expected_prediction(probabilities, categories):
    return sum(p * c for p, c in zip(probabilities, categories))

# Function to compute the entropy of a probability distribution
def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)

# Parameters
iterations = 1000
entropy_weight = 0.5  # Weighting factor for the entropy penalty

# Generate the probabilities (static over iterations)
probabilities = gaussian_probabilities(mean_category, std_dev, num_categories)
# probabilities = just_the_mean(mean_category, num_categories)
# probabilities = uniform(num_categories)

# Plot the distribution
ascii_plot(probabilities)

# Categories (0 to 10)
categories = list(range(num_categories))

# Sum the losses over 100 iterations
total_loss = 0

for _ in range(iterations):
    # Generate a random number (0 or 1)
    # actual_output = random.choice([0.0, 1.0])
    if random.randint(0,100) < 33:
        actual_output = 0
    else:
    	actual_output = 1
    # Compute the expected prediction
    expected_pred = expected_prediction(probabilities, categories)
    
    # Compute the absolute error loss
    error_loss = abs(expected_pred - actual_output)
    
    # Compute the entropy loss (to penalize spread-out distributions)
    entropy_loss = entropy(probabilities)
    
    # Total loss is a combination of error loss and entropy penalty
    loss = error_loss + entropy_weight * entropy_loss
    
    # Accumulate the loss
    total_loss += loss

# Calculate the average loss over iterations
average_loss = total_loss / iterations

# Display the average loss
print(f"Average Loss after {iterations} iterations: {average_loss}")
