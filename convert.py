import os
import sys
import argparse
from common import *

here = os.getcwd()
parser = argparse.ArgumentParser(description='Convert image datasets to FoCal (rank-order coded) spikes')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('input_dir', type=str, default=NO_IN_DIR)
parser.add_argument('--timestep', type=float, default=1.0)
parser.add_argument('--output_dir', type=str, default=os.path.join(here, 'output_spikes'))
parser.add_argument('--skip_existing', type=int, default=1)
parser.add_argument('--spikes_per_bin', type=int, default=1)
args = parser.parse_args()

def main():
    if not os.path.isdir(args.input_dir):
        raise Exception('Input directory not found')

    if args.dataset.lower() == CIFAR10.lower():
        import cifar_convert as cvt
    elif args.dataset.lower() == MNIST.lower():
        import mnist_convert as cvt
    else:
        raise Exception('Dataset not (yet) supported!')

    cvt.open_and_convert(args.input_dir, args.output_dir, 
        args.timestep, args.spikes_per_bin, args.skip_existing)


if __name__ == '__main__':
    main()
