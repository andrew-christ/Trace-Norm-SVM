import numpy as np

from sklearn.preprocessing import LabelEncoder

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


    
    def fit(self, X, y):
        """
        Fit the linear SVM model using training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.

        y : array-like of shape (n_samples,)
            Target labels relative to samples in X.

        Returns
        -------
        self : object
            Fitted estimator
        """

        n, self.d   = X.shape

        # Convert arbitrary labels to integers in range [0, k-1]
        y           = LabelEncoder().fit_transform(np.squeeze(y))

        # Number of classes
        self.k      = len(np.unique(y))

        # One-hot encoded labels for optimization
        Y           = np.zeros((n, self.k))
        Y[np.arange(n), y] = 1