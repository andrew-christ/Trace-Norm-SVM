import numpy as np

from sklearn.preprocessing import LabelEncoder

from tdqm import tqdm

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
        Fit a multi-class linear SVM model using an ADMM-based 
        optimization scheme.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data matrix.

        y : array-like of shape (n_samples,)
            Target labels relative to samples in X.

        Returns
        -------
        self : object
            Fitted estimator

        Notes
        -----
        The ADMM-based implementation follows the Crammer-Singer
        multi-class SVM formulation with hinge-loss constraints.
        """

        n, self.d   = X.shape

        # Convert arbitrary labels to integers in range [0, k-1]
        y           = LabelEncoder().fit_transform(np.squeeze(y))

        # Number of classes
        self.k      = y.max() + 1

        # One-hot encoded labels for optimization
        Y = np.zeros((n, self.k))
        Y[np.arange(n), y] = 1


        # ADMM primal Variables
        self.W      = np.zeros((self.d, self.k))    # Weights
        self.b      = np.zeros((1, self.k))         # Bias

        self.S      = np.zeros((n, self.k))         # Equality Constraint (Class Scores): S = WX + b
        self.M      = np.zeros((n, self.k))         # Equality Constraint (Margins): M = 1 + S - S_{y_i}

        # ADMM dual Variables
        self.U1     = np.zeros((n, self.k))
        self.U2     = np.zeros((n, self.k))

        self.loss_hist = []

        for _ in tqdm(range(self.max_iter), desc='Training SVM'):

            # 1.) M-Update

            # Select target class score: S_{y_i}
            S_hat   = self.S[np.arange(n), y][:, np.newaxis]

            # Compute margin violations
            V       = 1 - Y + self.S - S_hat - self.U1 / self.rho

            # Exclude target class from margin violations
            mask = np.ones_like(V, dtype=bool)
            mask[np.arange(n), y] = False
            V_masked = V[mask].reshape(n, self.k - 1)

            # Compute Crammer-Singer hinge proximal operator using dual
            lam = self.dual_simplex_projection(V_masked)

            # Primal update via KKT conditions of hinge proximal operator
            self.M = np.zeros_like(V)
            self.M[mask] = V[mask] - lam.ravel()