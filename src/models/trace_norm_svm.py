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


    def predict(self, X, return_scores=False):
        """
        Compute predictions using multi-class linear SVM.

        Class scores are of the form:

            S = XW + b

        where each entry S_{i,k} represents the score for
        class k for sample i.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data matrix.

        return_scores : bool, default=False
            If True, returns raw class scores instead of predicted labels

        Returns
        -------
        ndarray
            If return_scores=True:
                of shape (n_samples, n_classes)

            Otherwise:
                of shape (n_samples,)

        """

        if return_scores:
            return X @ self.W + self.b
        
        return self.le.inverse_transform(np.argmax(X @ self.W + self.b, axis=1))


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
        self.le     = LabelEncoder()
        y           = self.le.fit_transform(np.squeeze(y))

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
        self.T      = np.zeros((self.d, self.k))

        # ADMM dual Variables
        self.U1     = np.zeros((n, self.k))
        self.U2     = np.zeros((n, self.k))
        self.U3     = np.zeros((self.d, self.k))

        self.history = {
            "loss": [],
            "primal_residual": [],
            "dual_residual": []
        }

        for _ in tqdm(range(self.max_iter), desc='Training Trace Norm SVM'):

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
                np.eye(self.d) + X.T @ X,
                X.T @ (self.S + self.U2 / self.rho - self.b) + (self.T + self.U3 / self.rho)
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


            # 5.) T-Update

            self.T = self.nuclear_prox(self.W - self.U3 / self.rho)


            # 6.) U1-Update

            # Select target class score: S_{y_i}
            S_hat   = self.S[np.arange(n), y][:, np.newaxis]

            # Update Dual Variable
            self.U1 += self.rho * (self.M - (1 - Y + self.S - S_hat))


            # 7.) U2-Update

            # Update Dual Variable
            self.U2 += self.rho * (self.S - (X @ self.W + self.b))

            # 8.) U3-Update

            # Update Dual Variable
            self.U3 += self.rho * (self.T - self.W)


            # Append to loss history
            self.history["loss"].append(self._loss(X, y))

        return self


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
        sorting-based projection algorithm from  
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
        Solves the S-subproblem in the ADMM formulation of 
        Crammer-Singer SVM.

        The subproblem applies a class-dependent linear operator 
        that couples the class scores with the margin violations. 
        For each sample i, the update corresponds to the closed-form
        application of the inverse of:

            (I + B_{y_i}^T B_{y_i})^{-1}
        
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


    def _loss(self, X, y):
        """
        Compute the trace-norm multi-class SVM objective function.

        The loss is defined as:

            L(W) = ||W||_* + C * sum_i^n max_{k ≠ y_i} [1 + S_{i, k} - S_{i, y_i}]_+

        where:
            - S_{i, k} = (XW + b)_{i,k} is the score for class k
            - [·]_+ = max(·, 0) is the hinge function

        This corresponds to the multi-class hinge loss using the 
        Crammer-Singer formulation with trace-norm regularization.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix.

        y : ndarray of shape (n_samples,)
            Target labels relative to samples in X.

        Returns
        -------
        float
            Value of objective function (Trace-norm regularization + hinge loss)

        """

        n, d = X.shape

        # Class scores S = XW + b
        scores = X @ self.W + self.b

        # Extract correct class scores S_{i, y_i}
        correct_scores = scores[np.arange(n), y]

        # Compute margins: 1 + S_{i, k} - S_{i, y_i}
        margins = 1 + scores - correct_scores[:, np.newaxis]

        # Exclude correct class from loss
        margins[np.arange(n), y] = 0

        # Maximum hinge violation per sample
        hinge = np.maximum(margins, 0).max(axis=1)

        # Nuclear norm regularization term
        nuc_norm = np.linalg.norm(self.W, ord='nuc')

        return nuc_norm + self.C * hinge.sum()


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