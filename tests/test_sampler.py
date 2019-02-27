import pytest
import numpy as np
import numpy.testing as npt
import sample_generator as sg

def test_sampler_builds():
    R = np.diag([1., 1.])
    s = sg.SampleGenerator(covariance=R)
    return

def test_exception():
    with pytest.raises(Exception):
        s = sg.SampleGenerator()
