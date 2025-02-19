import numpy as np
import pytest
from data_structures import gradW, getPairwiseSeparations_inplace, gradW_inplace, gradW_float32

# Set up common test data
N = 400
h = 0.1
np.random.seed(42)
pos = np.random.randn(N, 3)
r = pos
M, N = pos.shape[0], pos.shape[0]

dx, dy, dz = getPairwiseSeparations_inplace(pos, pos)

dx_before = dx.copy()
dy_before = dy.copy()
dz_before = dz.copy()

dx_copy = dx.astype(np.float32, copy=True)
dy_copy = dy.astype(np.float32, copy=True)
dz_copy = dz.astype(np.float32, copy=True)

def test_xyz(): 
    assert np.allclose(dx, dx_copy, atol=1e-6), "dx changed unexpectedly!"
    assert np.allclose(dy, dy_before, atol=1e-6), "dy changed unexpectedly!"
    assert np.allclose(dz, dz_before, atol=1e-6), "dz changed unexpectedly!"

org_dWx, org_dWy, org_dWz = gradW(dx, dy, dz, h)
dWx, dWy, dWz = gradW_float32(dx_copy, dy_copy, dz_copy, h)

def test_dWx():
    assert org_dWx.shape == dWx.shape, "dx arrays have different shapes!"
    assert np.allclose(org_dWx, dWx, atol=1e-6), "dx values differ between methods!"

def test_dWy():
    assert org_dWy.shape == dWy.shape, "dy arrays have different shapes!"
    assert np.allclose(org_dWy, dWy, atol=1e-6), "dy values differ between methods!"

def test_dWz():
    assert org_dWz.shape == dWz.shape, "dz arrays have different shapes!"
    assert np.allclose(org_dWz, dWz, atol=1e-6), "dz values differ between methods!"