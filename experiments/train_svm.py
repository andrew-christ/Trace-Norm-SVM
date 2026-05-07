import argparse
import numpy

from src.models.svm import SVM

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

    clf = SVM(args.C, args.rho, args.max_iter)
    
    print("done")


if __name__ == "__main__":
    main()