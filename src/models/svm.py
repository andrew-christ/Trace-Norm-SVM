import numpy as np

class SVM:

    """Linear Multi-Class Support Vector Machine
    

    Parameters
    ----------
    C : float, default=0.1
        Regularization parameter. The strength of the regularization
        controls how strict the training examples must be correctly
        classified. Value must be strictly positive.

    rho : float, default=1
        Augmented Lagrangian parameter used in the ADMM implementation
        of the multi-class SVM.

        Larger values of `rho` enforce stricter adherence to the 
        primal equality constraints, but can hinder the convergence
        of the dual updates.

    max_iter : int, default=1000
        Maximum number of ADMM iterations
    
    """

    def __init__(self, C=0.1, rho=1, max_iter=1000):

        self.C              = C
        self.rho            = rho
        self.max_iter       = max_iter

        assert self.C > 0, 'Regularization parameter must be strictly positive'
        assert self.rho > 0, 'Augmented Lagrangian penalty parameter must be strictly positive'