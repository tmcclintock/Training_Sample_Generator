import numpy as np

class SampleGenerator(object):
    """
    Generate training samples for emulation
    """

    def __init__(self, chain=None, covariance=None, rotation_matrix=None):
        if (chain is None) and (covariance is None) and (rotation_matrix is None):
            raise Exception("Must supply either a chain, "+
                            "covariance, or rotation matrix.")
        return
