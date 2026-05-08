import argparse
import numpy
import matplotlib.pyplot as plt

from src.models.svm import SVM
from src.data.synthetic import SyntheticDataset
from src.visualization.decision_boundary import plot_boundary
from src.visualization.convergence import plot_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train Linear Multi-class SVM")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # 
    X, y = SyntheticDataset.linear()

    clf = SVM(args.C, args.rho, args.max_iter)
    clf.fit(X, y)

    plot_boundary(clf, X, y, ax[0, 0])
    plot_loss(clf, ax[0, 1])


    # 
    X, y = SyntheticDataset.gaussian(n_classes=5)

    clf = SVM(args.C, args.rho, args.max_iter)
    clf.fit(X, y)

    plot_boundary(clf, X, y, ax[1, 0])
    plot_loss(clf, ax[1, 1])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()