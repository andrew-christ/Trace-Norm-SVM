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
            lam     = self.dual_simplex_projection(V_masked)

            # Primal update via KKT conditions of hinge proximal operator
            self.M = np.zeros_like(V)
            self.M[mask] = V[mask] - lam.ravel()


            # 2.) W-Update
            self.W = np.linalg.solve(
                1 / self.rho * np.eye(self.d) + X.T @ X,
                X.T @ (self.S + self.U2 / self.rho - self.b)
            )


            # 3.) b-Update
            self.b = np.mean(self.S + self.U2 - X @ self.W, axis=0, keepdims=True)


            # 4.) S-Update

            # Margin violation with dual correction
            V = self.M + self.U1 / self.rho - 1 - Y

            # Linear model prediction with dual correction
            S = X @ self.W + self.b - self.U2 / self.rho

            # Removes target-class coupling contribution induced by margin constraints
            V_hat = np.zeros((n, self.k))
            V_hat[np.arange(n), y] = np.sum(V, axis=1)

            # Solve structured linear subproblem for S using closed-form operator
            self.S = self.solve_S_update(V - V_hat + S, y)


    def dual_simplex_projection(self, V):
        """
        Project the rows of a matrix V onto the simplex
        with radius 𝜏.

        Specifically, for each row v, compute the solution to:

            min_x   ||x - v||_2^2
            s.t.    x >= 0, sum(x) <= 𝜏

        where 𝜏 = C / rho. This corresponds to the dual of
        the hinge proximal operator

            min_{m, t}  ||m - v||_2^2 + 𝜏t
            s.t.        x_i <= t    ∀i

        which is used in this implementation of the
        Crammer-Singer hinge loss function.

        Parameters
        ----------
        V : ndarray of shape (n_samples, n_classes - 1)
            Matrix where rows correspond to margin 
            violation for non-target classes.
        
        Returns
        -------
        ndarray of shape (n_samples, n_classes - 1)
            Row-wise projection of V onto the simplex.
        
        Notes
        -----
        Row that already satisfy the simplex constraints 
        are left unchanged. This methods uses an efficient 
        sorting-based projection algorithm of  
        Duchi, John, et al. (2008)
        
        """

        n, k = V.shape

        # Upper bound of simplex
        tau     = self.C / self.rho

        # Threshold classes that already satisfy margin
        V_pos   = np.maximum(V, 0)

        # Identify rows that violate the simplex constraint (sum > 𝜏)
        mask    = (V_pos.sum(axis=1) > tau)[:, np.newaxis]

        # Sort each row in descending order
        U       = np.sort(V, axis=1)[:, ::-1]

        # Calculate cumulative sum of each sorted vector
        cssv    = np.cumsum(U, axis=1)

        # Condition that determines which elements are active
        ind     = np.arange(k) + 1
        cond    = U - (cssv - tau) / ind > 0

        # Largest index such that projection condition is satisfied
        p       = cond.sum(axis=1) - 1

        # Calculate the offset theta
        theta   = (cssv[np.arange(n), p] - tau) / (p + 1)
        theta   = theta[:, np.newaxis]

        # Shrink the rows that violate the simplex upper bound, then threshold
        return np.maximum(np.where(mask, V - theta, V), 0)
    
    def solve_S_update(self, Q, y):
        """
        Solve the S-subproblem in the ADMM formulation of 
        Crammer-Singer SVM.

        The subproblem applies a class-dependent linear operator 
        that couples the class scores with the margin violations. 
        For each sample i, the update corresponds to the closed-form
        application of the inverse of:

            (I + B_{y_i}^T B_{y_i})^{-1}

        where B_{y_i} = I + \mathbf{1} e_{y_i}^T. 
        
        This inverse operator can be computed in closed form 
        without explicity forming the matrix.

        Parameters
        ----------
        Q : ndarray of shape (n_samples, n_classes)
            Input matrix representing coupling of margin
            violations and class scores.

        y : array-like of shape (n_samples,)
            Target labels in {0, ..., k-1}, which determines the 
            structure of the inverse operator applied per row.

        Returns
        -------
        S : ndarray of shape (n_samples, n_classes)
            Updated S variable after solving structured subproblem.

        Notes
        -----
        Implementation avoids explicit matrix inversion for 
        efficiency. S is the auxiliary class-score variable
        introduced by ADMM splitting.
        
        """

        n, k = Q.shape

        # Sum over all classes
        q_sum = np.sum(Q, axis=1, keepdims=True)

        # Extract target class per sample
        q_y = Q[np.arange(n), y][:, np.newaxis]

        S = 0.5 * Q

        S += (q_y + q_sum) / (2 * (k+1))

        S[np.arange(n), y] += (q_sum[:, 0] - k * q_y[:, 0]) / (2 * (k+1))

        return S