import numpy as np

class SampleGenerator(object):
    """
    Generate training samples for emulation
    """

    def __init__(self, chain=None, covariance=None):
        if chain is not None:
            if covariance is not None:
                raise Exception("Must only supply one of a chain "+
                                "or covariance.")
            self.set_chain(chain)
        else:# covariance is not None
            self.set_covariance(covariance)
        self.set_scale(3)
        return

    def set_scale(self, scale):
        self.scale = scale
        return

    def set_chain(self, chain):
        chain = np.asarray(chain)
        if chain.ndim > 2:
            raise Exception("Chain must be at most a 2D array.")
        l1 = len(chain)
        l2 = len(chain[0])
        if l1 > l2:
            chain = chain.T
        self.chain = chain
        self.set_means(np.mean(chain, 0))
        self.set_covariance(np.cov(self.chain))
        return

    def set_means(self, means):
        self.means = means
        return
        
    def set_covariance(self, covariance):
        covariance = np.asarray(covariance)
        if covariance.ndim > 2:
            raise Exception("Covariance must be at most a 2D array.")
        self.covariance = covariance
        w, R = np.linalg.eig(self.covariance)
        self.eigenvalues = w
        self.rotation_matrix = R
        return

    def set_covariance_from_decomposition(self, eigenvalues,
                                          rotation_matrix):
        self.eigenvalues = eigenvalues
        self.rotation_matrix = rotation_matrix
        cov = np.zeros_like(self.rotation_matrix)
        for i in range(len(self.rotation_matrix)):
            cov += self.eigenvalues[i]*np.outer(self.rotation_matrix[:,i],
                                                self.rotation_matrix[:,i])
        self.covariance = cov
        return

    def set_seed(self, seed):
        np.random.seed(seed)
        return

    def generate_flat_samples(self, Nsamples):
        ndim = len(self.covariance)
        return np.random.random((Nsamples, ndim))

    def generate_circular_samples(self, Nsamples):
        theta = 2*np.pi*np.random.random(Nsamples)
        r = np.random.random(Nsamples) + np.random.random(Nsamples)
        r[(r>1)] = 2 - r[(r>1)]
        return np.array([r*np.cos(theta), r*np.sin(theta)])/2.
    
    
