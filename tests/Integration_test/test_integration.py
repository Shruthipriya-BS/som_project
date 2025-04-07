import os
import tempfile
import datetime
import numpy as np
import matplotlib.pyplot as plt
from kohonen_app import SOMAgent  

def test_small_som_image_generation():
    """
    Integration test for generating and saving a small SOM image.
    This test:
      - Creates a temporary directory for the images.
      - Trains a small SOM (10x10 grid, 100 iterations).
      - Saves the image in a subdirectory and verifies the file exists.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        small_dir = os.path.join(temp_dir, "images", "small")
        os.makedirs(small_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_data_small = np.random.random((10, 3))
        som_small = SOMAgent(width=10, height=10, input_dim=3, n_max_iterations=100)
        weights_small = som_small.train(input_data_small)
        filename = os.path.join(small_dir, f'100_{timestamp}.png')
        plt.imsave(filename, weights_small)
        # Check that the file was created and has non-zero size
        assert os.path.exists(filename), "Small SOM image was not saved."
        assert os.path.getsize(filename) > 0, "Small SOM image file is empty."

def test_large_som_image_generation():
    """
    Integration test for generating and saving a large SOM image.
    This test:
      - Creates a temporary directory for the images.
      - Trains a large SOM (100x100 grid, 1000 iterations).
      - Saves the image in a subdirectory and verifies the file exists.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        large_dir = os.path.join(temp_dir, "images", "large")
        os.makedirs(large_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_data_large = np.random.random((10, 3))
        som_large = SOMAgent(width=100, height=100, input_dim=3, n_max_iterations=1000)
        weights_large = som_large.train(input_data_large)
        filename = os.path.join(large_dir, f'1000_{timestamp}.png')
        plt.imsave(filename, weights_large)
        # Check that the file was created and has non-zero size
        assert os.path.exists(filename), "Large SOM image was not saved."
        assert os.path.getsize(filename) > 0, "Large SOM image file is empty."
