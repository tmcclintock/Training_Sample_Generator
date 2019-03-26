import pytest
import numpy as np
import numpy.testing as npt
import sample_generator as sg

def test_sampler_builds(): #smoke test
    R = np.diag([1., 1.])
    s = sg.SampleGenerator(covariance=R)
    s = sg.SampleGenerator(covariance=R, scale=5)

    chain = np.random.randn(2, 1000)
    s = sg.SampleGenerator(chain=chain)
    s = sg.SampleGenerator(chain=chain, scale=5)
    return

def test_exceptions():
    R = np.diag([1., 1.])
    chain = np.random.randn(2, 1000)
    with pytest.raises(Exception):
        s = sg.SampleGenerator()
    with pytest.raises(Exception):
        s = sg.SampleGenerator(chain=chain, scale=0)
    with pytest.raises(Exception):
        s = sg.SampleGenerator(chain=chain, scale=-1)
    with pytest.raises(Exception):
        s = sg.SampleGenerator(covariance=R, scale=0)
    with pytest.raises(Exception):
        s = sg.SampleGenerator(covariance=R, scale=-1)
    with pytest.raises(Exception):
        s = sg.SampleGenerator(covariance=R, scale=99)
    return
