import numpy as np
import pytest
from data_structures import getPairwiseSeparations, getPairwiseSeparations_inplace

# Set up common test data
N = 400
np.random.seed(42)
pos = np.random.randn(N, 3)
r = pos
M, N = pos.shape[0], pos.shape[0]
dx = np.empty((M, N), dtype=np.float32)
dy = np.empty((M, N), dtype=np.float32)
dz = np.empty((M, N), dtype=np.float32)

# Compute separations using both methods
org_dx, org_dy, org_dz = getPairwiseSeparations(r, pos)
dx, dy, dz = getPairwiseSeparations_inplace(r, pos, dx, dy, dz)

def test_dx():
    assert org_dx.shape == dx.shape, "dx arrays have different shapes!"
    assert np.allclose(org_dx, dx, atol=1e-6), "dx values differ between methods!"

def test_dy():
    assert org_dy.shape == dy.shape, "dy arrays have different shapes!"
    assert np.allclose(org_dy, dy, atol=1e-6), "dy values differ between methods!"

def test_dz():
    assert org_dz.shape == dz.shape, "dz arrays have different shapes!"
    assert np.allclose(org_dz, dz, atol=1e-6), "dz values differ between methods!"