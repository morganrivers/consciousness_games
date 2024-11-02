import math
import random

# Number of categories (0 to 10)
num_categories = 11

def ascii_plot(bin_sums, max_height=20):
    max_value = max(bin_sums)
    if max_value == 0:
        max_value = 1
    heights = [int((value / max_value) * max_height) for value in bin_sums]
    for i, height in enumerate(heights):
        p_value = i / 10  # Convert bin index to p value
        print(f"p = {p_value:3.1f}: " + '|' * height)

# # Generate probabilities using a Gaussian distribution centered around a mean and std_dev
# def gaussian_probabilities(mean, std_dev, num_categories):
#     probabilities = [math.exp(-0.5 * ((i - mean) / std_dev) ** 2) for i in range(num_categories)]
#     sum_probabilities = sum(probabilities)
    
#     # # Avoid division by zero by adding a small epsilon
#     # if sum_probabilities == 0:
#     #     sum_probabilities += epsilon
    
#     probabilities = [p / sum_probabilities for p in probabilities]

# Generate static probabilities using a Gaussian distribution centered around category 5
def gaussian_probabilities(mean, std_dev, num_categories):
    print()
    print(f"std_dev {std_dev}")
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

# Function to compute the loss
def compute_loss(mean, std_dev, categories, actual_output, entropy_weight=0.5):
    
    # Compute expected prediction
    expected_pred = expected_prediction(probabilities, categories)
    
    # Compute the error loss
    error_loss = abs(expected_pred - actual_output)
    
    # Compute the entropy loss
    entropy_loss = entropy(probabilities)
    
    # Total loss
    total_loss = error_loss + entropy_weight * entropy_loss
    return total_loss

# Hill climbing learner
def hill_climb_optimizer(iterations, entropy_weight=0.5, learning_rate=0.1, decay=0.95, min_lr=0.001, min_std_dev=0.001):
    mean = random.uniform(0, num_categories)
    std_dev = random.uniform(0.5, 3.0)
    
    mean_lr = learning_rate
    std_dev_lr = learning_rate

    categories = list(range(num_categories))

    for i in range(iterations):
        actual_output = 0 if random.randint(0, 100) < 33 else 1

        current_loss = compute_loss(mean, std_dev, categories, actual_output, entropy_weight)

        mean_new = mean + mean_lr
        std_dev_new = std_dev + std_dev_lr

        # Ensure standard deviation is always positive
        std_dev_new = max(std_dev_new, min_std_dev)

        new_loss = compute_loss(mean_new, std_dev_new, categories, actual_output, entropy_weight)

        if new_loss < current_loss:
            mean = mean_new
            std_dev = std_dev_new
        else:
            mean_lr = max(mean_lr * -decay, min_lr)  # Use minimum learning rate
            std_dev_lr = max(std_dev_lr * -decay, min_lr)

        # Optionally print progress every few iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Mean: {mean}, Std Dev: {std_dev}, Loss: {current_loss}")
    
    return mean, std_dev

# Parameters
iterations = 1000
entropy_weight = 0.5
learning_rate = 0.1

# Run the optimizer
final_mean, final_std_dev = hill_climb_optimizer(iterations, entropy_weight, learning_rate)

# Display final results
print(f"Final Mean: {final_mean}, Final Std Dev: {final_std_dev}")
