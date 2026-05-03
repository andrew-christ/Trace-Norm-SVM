import numpy as np

class SVM:

    """Linear Support Vector Machine
    

    Parameters
    ----------
    C : float, default=0.1
        Regularization parameter. The strength of the regularization
        controls how strict the training examples must be correctly
        classified. Value must be strictly positive.
    
    """

    def __init__(self, C=0.1, rho=1, max_iter=1000):

        self.C              = C
        self.rho            = rho
        self.max_iter       = max_iter

        assert self.C > 0, 'Regularization parameter must be strictly positive'