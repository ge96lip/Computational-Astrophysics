import numpy as np
import pytest
import opt_sph
import sph

# Set up common test data
N = 1000
h = 0.1
np.random.seed(42)
pos = np.random.randn(N, 3)
r = pos
M, N = pos.shape[0], pos.shape[0]

opt_dx, opt_dy, opt_dz = opt_sph.getPairwiseSeparations(pos, pos)
dx, dy, dz = sph.getPairwiseSeparations(pos, pos)


# Test
def test_dx():
    assert opt_dx.shape == dx.shape, "dx shape mismatch"
    assert np.allclose(opt_dx, dx, rtol=1e-5, atol=1e-8), "dx values mismatch"


def test_dy():
    assert opt_dy.shape == dy.shape, "dy shape mismatch"
    assert np.allclose(opt_dy, dy, rtol=1e-5, atol=1e-8), "dy values mismatch"


def test_dz():
    assert opt_dz.shape == dz.shape, "dz shape mismatch"
    assert np.allclose(opt_dz, dz, rtol=1e-5, atol=1e-8), "dz values mismatch"
