import numpy as np
import matplotlib.pyplot as plt


def _make_grid(X, resolution=500, padding=1.5):
    """
    Create a 2D mesh grid covering the data range.
    """
    mu = np.mean(X, axis=0)
    t = np.max(np.abs(X - mu)) * padding

    xx = np.linspace(mu[0] - t, mu[0] + t, resolution)
    yy = np.linspace(mu[1] - t, mu[1] + t, resolution)

    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]

    return XX, YY, grid

def _plot_background(ax, XX, YY, Z, cmap):
    """
    Plot decision regions.
    """
    contour = ax.contourf(XX, YY, Z, alpha=0.25, cmap=cmap)
    return contour

def _plot_points(ax, X, y, cmap):
    """
    Scatter plot of training/test points.
    """
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap,
        edgecolors="k",
        s=30
    )

def plot_boundary(model, X, y, ax=None, resolution=500, use_scores=False, cmap="Spectral"):
    """
    Wrapper for decision boundary visualization.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    cmap = plt.colormaps.get_cmap(cmap)

    # 1. Make grid for decision boundary
    XX, YY, grid = _make_grid(X, resolution=resolution)

    # 2. Make predictions using model
    Z = model.predict(grid)
    Z = Z.reshape(XX.shape)

    # 3. Plot decision regions
    contour = _plot_background(ax, XX, YY, Z, cmap)

    # 4. Plot data points on decision boundary
    _plot_points(ax, X, y, cmap)

    fig.colorbar(contour, ax=ax)
    ax.set_title("Decision Boundary")

    return fig, ax