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

dx, dy, dz = sph.getPairwiseSeparations(pos, pos)

org_dWx, org_dWy, org_dWz = sph.gradW(dx, dy, dz, h)
dWx, dWy, dWz = opt_sph.gradW(dx, dy, dz, h)


# Test
def test_dWx():
    assert org_dWx.shape == dWx.shape, "dx shape mismatch"
    assert np.allclose(org_dWx, dWx, rtol=1e-5, atol=1e-8), "dx values mismatch"


def test_dWy():
    assert org_dWy.shape == dWy.shape, "dy shape mismatch"
    assert np.allclose(org_dWy, dWy, rtol=1e-5, atol=1e-8), "dy values mismatch"


def test_dWz():
    assert org_dWz.shape == dWz.shape, "dz shape mismatch"
    assert np.allclose(org_dWz, dWz, rtol=1e-5, atol=1e-8), "dz values mismatch"
