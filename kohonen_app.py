# kohonen_app.py

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import logging

# Set Matplotlib to use the non-interactive Agg backend for image generation
import matplotlib
matplotlib.use("Agg")

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="training.log",  # Write logs to training.log
    filemode="a"              # Append mode
)
logger = logging.getLogger(__name__)

# -------------------------------
# Define the SOMAgent class for SOM training
# -------------------------------
class SOMAgent:
    def __init__(self, width: int, height: int, input_dim: int, n_max_iterations: int, alpha0: float = 0.1):
        """
        Initializes a new instance of SOMAgent with a given grid size (width and height), 
        the dimensionality of the input vector (input_dim), maximum iterations, and an initial learning rate (alpha0).
        """
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.n_max_iterations = n_max_iterations
        self.alpha0 = alpha0
        self.sigma0 = max(width, height) / 2.0
        self.weights = np.random.random((width, height, input_dim))
        # Time constant for decay (λ)
        self.lmbda = n_max_iterations / np.log(self.sigma0)
        # Precompute grid coordinates (an array of shape (width, height, 2))
        self.coords = np.indices((width, height)).transpose(1, 2, 0)

    def find_bmu(self, vt: np.ndarray) -> tuple:
        """
        Calculates the Euclidean distance between the current input vector vt and all nodes weight vectors.
        Finds the Best Matching Unit (BMU)—the node whose weights are closest to vt.
        Returns the BMU's coordinates as a tuple.
        """
        distances = np.linalg.norm(self.weights - vt, axis=2)
        bmu_idx = np.argmin(distances)
        bmu_x, bmu_y = np.unravel_index(bmu_idx, (self.width, self.height))
        return (bmu_x, bmu_y)

    def update_weights(self, vt: np.ndarray, bmu: tuple, alpha_t: float, sigma_t: float):
        """
        Computes the distance from each node to the BMU using precomputed grid coordinates.
        Calculates the influence of the BMU on each node using an exponential decay function.
        Updates the weights of all nodes in a vectorized manner, bringing them closer to the input vector vt based on the calculated influence and 
        current learning rate (alpha_t).
        """
        bmu_coords = np.array(bmu)
        # Vectorized Euclidean distances from each node to the BMU:
        dists = np.linalg.norm(self.coords - bmu_coords, axis=2)
        # Calculate the influence of the BMU on all nodes at once:
        influence = np.exp(-(dists ** 2) / (2 * (sigma_t ** 2)))
        # Update the weights in a vectorized manner:
        self.weights += alpha_t * influence[..., np.newaxis] * (vt - self.weights)

    def train(self, input_data: np.ndarray) -> np.ndarray:
        """
        Runs the training loop for a fixed number of iterations.
        In each iteration: Computes the current neighborhood radius (sigma_t) and learning rate (alpha_t) using exponential decay.
        Iterates over each input vector (vt) in the training data:
        Finds the BMU for vt.
        Updates the weights of all nodes based on the BMU and the current values of alpha_t and sigma_t.
        Logs progress every 10 iterations.
        Returns the final weight matrix after training.
        """
        learning_rates = []
        for t in range(self.n_max_iterations):
            sigma_t = self.sigma0 * np.exp(-t / self.lmbda)
            alpha_t = self.alpha0 * np.exp(-t / self.lmbda)
            learning_rates.append(alpha_t)
            for vt in input_data:
                bmu = self.find_bmu(vt)
                self.update_weights(vt, bmu, alpha_t, sigma_t)
            if t % 10 == 0:
                logger.info(f"Iteration {t}/{self.n_max_iterations} complete.")
        # Ensure the final iteration is logged if not already done
        if (self.n_max_iterations - 1) % 10 != 0:
            logger.info(f"Iteration {self.n_max_iterations}/{self.n_max_iterations} complete.")
        return self.weights


if __name__ == '__main__':
    """
    Main entry point for the application. 
    It creates directories for saving images,generates sample data, trains the SOM, and saves the resulting images.
    The images are saved in two directories: 'images/small' and 'images/large'
    """
    # Create directories if they don't exist
    small_dir = os.path.join("images", "small")
    large_dir = os.path.join("images", "large")
    os.makedirs(small_dir, exist_ok=True)
    os.makedirs(large_dir, exist_ok=True)
    
    # Generate a timestamp for the image filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train and save a small SOM image (10x10 grid, 100 iterations)
    input_data_small = np.random.random((10, 3))
    som_small = SOMAgent(width=10, height=10, input_dim=3, n_max_iterations=100)
    weights_small = som_small.train(input_data_small)
    plt.imsave(os.path.join(small_dir, f'100_{timestamp}.png'), weights_small)
    
    # Train and save a large SOM image (100x100 grid, 1000 iterations)
    input_data_large = np.random.random((10, 3))
    som_large = SOMAgent(width=100, height=100, input_dim=3, n_max_iterations=1000)
    weights_large = som_large.train(input_data_large)
    plt.imsave(os.path.join(large_dir, f'1000_{timestamp}.png'), weights_large)
    
