import numpy as np

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm


class TraceNormSVM:

    """
    Linear Multi-Class Support Vector Machine 
    with Trace Norm Regularization


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

        self.C          = C
        self.rho        = rho
        self.max_iter   = int(max_iter)

        assert self.C > 0, 'Regularization parameter must be strictly positive'
        assert self.rho > 0, 'Augmented Lagrangian penalty parameter must be strictly positive'





    def nuclear_prox(self, Z):
        """
        Proximal operator of the nuclear norm via singular value
        thresholding (SVT).

        Solves the optimization problem:

            minimize_X  ||X||_* + (rho / 2) * ||X - Z||_F^2

        where:
            - ||⋅||_* is the nuclear norm (sum of singular values)
            - ||⋅||_F is the Frobenius norm
            - rho > 0 is the regularization parameter

        This has a closed-form solution given by the 
        soft-thresholding of the singular values of Z.

        Parameters
        ----------
        Z : ndarray of shape (n_features, n_classes)
            Input matrix.

        Returns
        -------
        X : ndarray of shape (n_features, n_classes)
            Solution of proximal problem

        Notes
        -----
        This is also known as Singular Value Thresholding (SVT) [1],
        commonly used in matrix completion and low-rank regularization.

        References
        ----------
            [1] Cai, Jian-Feng, Emmanuel J. Candès, and Zuowei Shen. 
                "A singular value thresholding algorithm for matrix completion." 
                SIAM Journal on optimization 20.4 (2010): 1956-1982

        """

        # Compute the compact singular value decomposition (SVD)
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)

        # Apply soft-threholding to the singular values.
        s_thresh = np.maximum(s - 1.0 / self.rho, 0)

        # Reconstruct the matrix using the thresholded singular values
        return U @ np.diag(s_thresh) @ Vt