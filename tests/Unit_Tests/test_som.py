import numpy as np
import pytest
from kohonen_app import SOMAgent

def test_find_bmu_returns_valid_index():
    agent = SOMAgent(width=5, height=5, input_dim=3, n_max_iterations=10)
    # Set weights to zeros so that any input vector has equal distance.
    agent.weights = np.zeros((5, 5, 3))
    vt = np.array([1.0, 1.0, 1.0])
    bmu = agent.find_bmu(vt)
    assert isinstance(bmu, tuple)
    assert len(bmu) == 2
    assert 0 <= bmu[0] < 5
    assert 0 <= bmu[1] < 5

def test_update_weights_changes_weights():
    agent = SOMAgent(width=5, height=5, input_dim=3, n_max_iterations=10)
    vt = np.array([1.0, 1.0, 1.0])
    bmu = (2, 2)
    original_weights = agent.weights.copy()
    agent.update_weights(vt, bmu, alpha_t=0.5, sigma_t=1.0)
    assert not np.array_equal(original_weights, agent.weights)

