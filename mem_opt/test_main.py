import pytest
import opt_sph
import sph

# REMEMBER TO REMOVE @PROFILE


def test_main():
    # test the sph .py file
    assert sph.main() == 0
