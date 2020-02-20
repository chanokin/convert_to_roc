import os
import sys
import argparse
from constants import DATASETS, NO_IN_DIR, CIFAR10, MNIST, OMNIGLOT

here = os.getcwd()
parser = argparse.ArgumentParser(description='Convert image datasets to FoCal (rank-order coded) spikes')
parser.add_argument('dataset', type=str, choices=DATASETS, 
                    help='Name of the dataset to convert')
parser.add_argument('input_dir', type=str, default=NO_IN_DIR, 
                    help='Path to the directory containing (standard) dataset files')
parser.add_argument('--timestep', type=float, default=1.0, 
                    help='Timestep which will be used in the simulations. How many spikes'
                    ' will be emmited at each timestep can be set with --spikes_per_bin')

parser.add_argument('--percent', type=float, default=0.3, 
                    help='How many of the possible spikes (number of pixels) should we '
                    ' output. Percent (0.0 < p <= 1.0)')

parser.add_argument('--output_dir', type=str, default=os.path.join(here, 'output_spikes'),
                    help='Path to the output location of the generated spike files')
parser.add_argument('--skip_existing', type=int, default=1,
                    help='Whether to skip database entries corresponding to files already'
                    ' found in the output directory')
parser.add_argument('--spikes_per_bin', type=int, default=1,
                    help='How many spikes per timestep will be emmited. Note that more '
                    ' than one is not standard rank-order encoding.')
parser.add_argument('--scaling', type=float, default=0.54,
                    help='Scaling applied to the input image (only supported by the '
                    ' Omniglot dataset)')
args = parser.parse_args()

def main():
    if not os.path.isdir(args.input_dir):
        raise Exception('Input directory not found')

    if args.dataset.lower() == CIFAR10.lower():
        import cifar_convert as cvt
    elif args.dataset.lower() == MNIST.lower():
        import mnist_convert as cvt
    elif args.dataset.lower() == OMNIGLOT.lower():
        import omniglot_convert as cvt
    else:
        raise Exception('Dataset not (yet) supported!')

    cvt.open_and_convert(args.input_dir, args.output_dir, args.percent,
        args.timestep, args.spikes_per_bin, args.skip_existing, args.scaling)


if __name__ == '__main__':
    main()
