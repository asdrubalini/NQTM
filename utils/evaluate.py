import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theta_path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    theta = np.load(args.theta_path)

    print("Theta matrix shape:", theta.shape)
    print("Theta matrix shape:", theta[0])
