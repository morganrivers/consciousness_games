import math
import random
from time import sleep
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


# Generate static probabilities using a Gaussian distribution centered around category 5
def gaussian_probabilities(mean, std_dev, num_categories):
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
    return -sum(p * math.log(p + 1e-8) for p in probabilities if p > 0)

# Function to compute the loss
def compute_loss(probabilities, categories, actual_output, entropy_weight=0.1):

    ascii_plot(probabilities)
    
    # Compute expected prediction
    expected_pred = expected_prediction(probabilities, categories)
    # print("expected_pred")
    # print(expected_pred)
    # Compute the error loss
    error_loss = abs(expected_pred - actual_output)
    print("error_loss")
    print(error_loss)
    # Compute the entropy loss
    entropy_loss = entropy(probabilities)
    # print("entropy_loss")
    # print(entropy_loss)
    # Total loss
    total_loss = error_loss + entropy_weight * entropy_loss
    # print(f"total_loss {total_loss}")
    return total_loss

# A very simple neural network-like model to optimize the mean and std_dev
class SimpleNeuralNetwork:
    def __init__(self, mean_init, std_dev_init, learning_rate):
        self.mean = mean_init
        self.std_dev = std_dev_init
        self.learning_rate = learning_rate

    # Forward pass to compute loss
    def forward(self, actual_output, categories, entropy_weight):
        probabilities = gaussian_probabilities(self.mean, self.std_dev, len(categories))
        return compute_loss(probabilities, categories, actual_output, entropy_weight)

    # Gradient descent step to update mean and std_dev
    def backward(self, actual_output, categories, entropy_weight):
        loss = self.forward(actual_output, categories, entropy_weight)
        
        # Compute gradients via numerical approximation (finite differences)
        delta = 1e-5

        probabilities_higher_mean = gaussian_probabilities(self.mean + delta, self.std_dev, len(categories))
        mean_grad = (compute_loss(probabilities_higher_mean, categories, actual_output, entropy_weight) - loss) / delta

        probabilities_wider = gaussian_probabilities(self.mean, self.std_dev + delta, len(categories))
        std_dev_grad = (compute_loss(probabilities_wider, categories, actual_output, entropy_weight) - loss) / delta

        # Gradient descent update for mean and std_dev
        self.mean -= self.learning_rate * mean_grad
        self.std_dev -= self.learning_rate * std_dev_grad
        
        # Ensure std_dev stays positive
        self.std_dev = max(self.std_dev, 0.001)
        
        return loss



# Categories (0 to 10)
categories = list(range(num_categories))

# probabilities = just_the_mean(mean_category, num_categories)
probabilities = uniform(num_categories)
def get_sample_losses(actual_output):
    print("\n\nactual_output")
    print(actual_output)
    probabilities = uniform(11)
    loss = compute_loss(probabilities, categories, actual_output, entropy_weight=0.1)
    print(f"uniform loss: {loss}")
    probabilities = just_the_mean(5,11)
    loss = compute_loss(probabilities, categories, actual_output, entropy_weight=0.1)
    print(f"just_the_mean loss: {loss}")
    probabilities = gaussian_probabilities(5, 2, 11)
    loss = compute_loss(probabilities, categories, actual_output, entropy_weight=0.1)
    print(f"gaussian_probabilities loss: {loss}")
get_sample_losses(0)
get_sample_losses(10)
quit()
print("")
print("")
print("")
print("")
print("training a NN")
# Parameters for the neural network and optimization
iterations = 1000000
learning_rate = 0.1
entropy_weight = 0.01#0.01#0.1

# Initialize the neural network with random mean and std_dev
nn = SimpleNeuralNetwork(mean_init=random.uniform(0, num_categories), std_dev_init=random.uniform(0.5, 3.0), learning_rate=learning_rate)

# Training loop
total_loss = 0
for i in range(iterations):
    # Generate a random actual output (0 or 1)
    actual_output = 0 if random.randint(0, 100) < 33 else 10
    # actual_output = 5
    # actual_output = 10*random.randint(0,1)
    # get_sample_losses(0)
    # get_sample_losses(1)
    # get_sample_losses(10)
    # loss = compute_loss(probabilities, categories, actual_output, entropy_weight=0.1)
    # Perform a backward pass (which computes the gradients and updates the weights)
    loss = nn.backward(actual_output, categories, entropy_weight)
    # total_loss += loss
    #     sleep(1)

    #     # Optionally print progress every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}, Mean: {nn.mean}, Std Dev: {nn.std_dev}, Loss: {loss}")

# Final results
print(f"Final Mean: {nn.mean}, Final Std Dev: {nn.std_dev}, Average loss over entire: {total_loss/iterations}")
