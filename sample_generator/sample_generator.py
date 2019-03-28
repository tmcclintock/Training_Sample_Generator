import numpy as np

class SampleGenerator(object):
    """
    Generate training samples for emulation
    """

    def __init__(self, chain=None, covariance=None, scale=3):
        self.set_scale(scale)
        if chain is not None:
            if covariance is not None:
                raise Exception("Must only supply one of a chain "+
                                "or covariance.")
            self.set_chain(chain)
        else:# covariance is not None
            self.set_means(np.zeros_like(covariance[0]))
            self.set_covariance(covariance)
        return

    def set_scale(self, scale):
        if scale <= 0:
            raise Exception("Scale cannot be <=0.")
        if scale > 15:
            raise Exception("Scale value is unrealistically large.")
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
        self.set_means(np.mean(chain, 1))
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
        self.transformation_matrix = self.scale*np.dot(R, np.sqrt(w))
        return

    def set_covariance_from_decomposition(self, eigenvalues,
                                          rotation_matrix):
        self.eigenvalues = eigenvalues
        self.rotation_matrix = rotation_matrix
        w = eigenvalues
        R = rotation_matrix
        self.transformation_matrix = self*scale*np.dot(np.sqrt(w), R)

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
        return np.array([r*np.cos(theta), r*np.sin(theta)]).T/2. + 0.5

    def generate_grid_samples(self, Nsamples):
        ndim = len(self.covariance)
        samples = np.random.randint(0, Nsamples-1, size=(Nsamples, ndim))
        return samples.astype(float)/Nsamples
    
    def generate_LHMDU_samples(self, Nsamples, oversampling=5,
                               Nneighbors=2):
        N = Nsamples*oversampling
        ndim = len(self.covariance)
        flat_rands = self.generate_flat_samples(N)
        #Compute distances beteween all random points
        distances = np.zeros((N, N))
        for i in range(0, N):
            for j in range(i+1, N):
                distances[i,j] = distances[j,i] = \
                    np.linalg.norm(flat_rands[i]-flat_rands[j])
                continue
            continue
        
        #Eliminate points based on those that have minimum distances to other
        #points, until we are down to Nsamples (the number we want)
        aveDistances = np.zeros(N)
        indices = np.arange(N)
        while(len(aveDistances) > Nsamples):
            for i, row in enumerate(sorted(indices)):
                meanAveDist = \
                    np.mean(sorted(distances[i,sorted(indices)])[:Nneighbors+1])
                #Note: the +1 is "to remove the zero index"
                #Append the average distances
                aveDistances[i] = meanAveDist
                continue
            #Delete the row with the minimum mean distance
            index_to_delete = np.argmin(aveDistances)
            indices = np.delete(indices, index_to_delete)
            aveDistances = np.delete(aveDistances, index_to_delete)
            continue
        
        #The output matrix
        strata_matrix = flat_rands[sorted(indices)]
        if len(strata_matrix) != Nsamples:
            raise Exception("Number of samples in the strata matrix is not "+\
                            "the requested amount.")
        if len(strata_matrix[0]) != ndim:
            raise Exception("Dimensions of strata matrix is not the same "+\
                            "as your covariance.")

        return strata_matrix
        
    def get_samples(self, Nsamples, method="flat", **kwargs):
        if method=="flat":
            x = self.generate_flat_samples(Nsamples)
        elif method=="circular":
            x = self.generate_circular_samples(Nsamples)
        elif method=="LHMDU":
            items = kwargs.items()
            oversampling = 5
            Nneighbors = 2
            if bool(items):
                if "oversampling" in items.keys():
                    oversampling = items["oversampling"]
                if "Nneighbors" in items.keys():
                    Nneighbors = items["Nneighbors"]
                for key, _ in items:
                    if key not in ["oversampling", "Nneighbors"]:
                        print("Keyword %s does nothing."%(key))
            x = self.generate_LHMDU_samples(Nsamples, oversampling, Nneighbors)
        else:
            raise Exception("Invalid sample generation method.")
        self.current_projected_samples = x
        x -= 0.5 #center
        s = self.scale
        w = self.eigenvalues
        R = self.rotation_matrix
        return np.dot(s*x[:]*np.sqrt(w), R.T)[:] + self.means
        
if __name__ == "__main__":
    c = np.diag([1,1])
    sg = SampleGenerator(covariance=c)
    x = sg.get_samples(10, "flat")
