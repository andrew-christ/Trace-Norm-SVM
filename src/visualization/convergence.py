import numpy as np
import matplotlib.pyplot as plt


def plot_loss(model, ax=None, log_scale=False):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure


    loss = np.array(model.history["loss"])

    ax.plot(loss, label="Loss", linewidth=2)

    if log_scale:
        ax.set_yscale("log")

    ax.set_title("ADMM SVM Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")

    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax