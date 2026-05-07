import numpy as np


class SyntheticDataset:

    @staticmethod
    def gaussian(n_classes=3, n_samples=50, seed=0):
        np.random.seed(seed)

        X, y = [], []

        cov = np.array([[0.1, 0], [0, 0.1]])

        for k in range(n_classes):
            mean = np.array([
                np.cos(2 * np.pi * k / n_classes),
                np.sin(2 * np.pi * k / n_classes)
            ])

            Xk = np.random.multivariate_normal(mean, cov, n_samples)
            yk = np.full(n_samples, k)

            X.append(Xk)
            y.append(yk)

        return np.vstack(X), np.hstack(y)


    @staticmethod
    def linear(n_samples=50, n_classes=3, seed=0):

        np.random.seed(seed)

        cov = np.array([
            [1, 0.1],
            [0.1, 1]
        ])

        base_mu = np.array([
            [0, 3],
            [0, 0],
            [0, -3]
        ], dtype=np.float32)

        mu = base_mu + np.random.normal(scale=0.5, size=base_mu.shape)

        X, y = [], []

        for i, m in enumerate(mu[:n_classes]):
            Xi = np.random.multivariate_normal(m, cov, n_samples)
            yi = np.full(n_samples, i)

            X.append(Xi)
            y.append(yi)

        return np.vstack(X), np.hstack(y)