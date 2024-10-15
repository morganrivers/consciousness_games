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

# Generate static probabilities using a Gaussian distribution centered around category 5
def gaussian_probabilities(mean, std_dev, num_categories):
    probabilities = [math.exp(-0.5 * ((i - mean) / std_dev) ** 2) for i in range(num_categories)]
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
    error_loss = abs(expected_pred - actual_output)
    
    entropy_loss = entropy(probabilities)
    total_loss = error_loss + entropy_weight * entropy_loss
    return total_loss

# Adam optimizer implementation
class AdamOptimizer:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        update = self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
        return update

# A simple neural network to optimize the mean and std_dev using Adam
class SimpleNeuralNetwork:
    def __init__(self, mean_init, std_dev_init, learning_rate):
        self.mean = mean_init
        self.std_dev = std_dev_init
        self.min_std_dev = 3
        self.max_std_dev = 10
        self.min_mean = 0
        self.max_mean = 10
        self.learning_rate = learning_rate
        self.mean_optimizer = AdamOptimizer(learning_rate)
        self.std_dev_optimizer = AdamOptimizer(learning_rate)

    # Forward pass to compute loss
    def forward(self, actual_output, categories, entropy_weight):
        probabilities = gaussian_probabilities(self.mean, self.std_dev, len(categories))
        return compute_loss(probabilities, categories, actual_output, entropy_weight)

    # Analytical gradients and backward pass to update mean and std_dev
    def backward(self, actual_output, categories, entropy_weight):
        print("self.std_dev")
        print(self.std_dev)
        probabilities = gaussian_probabilities(self.mean, self.std_dev, len(categories))
        loss = compute_loss(probabilities, categories, actual_output, entropy_weight)
        
        # Compute gradients analytically
        mean_grad = sum(prob * (i - self.mean) / (self.std_dev ** 2) for prob, i in zip(probabilities, categories))
        std_dev_grad = sum(prob * (((i - self.mean) ** 2 - self.std_dev ** 2) / (self.std_dev ** 3)) for prob, i in zip(probabilities, categories))
        
        # Clip gradients
        mean_grad = max(min(mean_grad, 1.0), -1.0)
        std_dev_grad = max(min(std_dev_grad, 1.0), -1.0)
        
        # Update mean and std_dev using Adam optimizer
        self.mean -= self.mean_optimizer.update(mean_grad)
        self.std_dev -= self.std_dev_optimizer.update(std_dev_grad)
        
        # Ensure std_dev stays positive
        self.std_dev = max(self.std_dev, self.min_std_dev)
        
        return loss

# Categories (0 to 10)
categories = list(range(num_categories))

# Training parameters
iterations = 10000
learning_rate = 0.01
entropy_weight = 0#.1

# Initialize the neural network with better initial values
nn = SimpleNeuralNetwork(mean_init=num_categories / 2, std_dev_init=3.0, learning_rate=learning_rate)

# Training loop
for i in range(iterations):
    # Generate a random actual output (0 or 10)
    actual_output = 0 if random.random() < 0.33 else 1

    # Perform a backward pass (which computes the gradients and updates the weights)
    loss = nn.backward(actual_output, categories, entropy_weight)
    from time import sleep
    sleep(0.001)

    # Optionally print progress every 1000 iterations
    if i % 1000 == 0:
        print(f"Iteration {i}, Mean: {nn.mean}, Std Dev: {nn.std_dev}, Loss: {loss}")

# Final results
print(f"Final Mean: {nn.mean}, Final Std Dev: {nn.std_dev}")
