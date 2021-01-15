import numpy as np
import sys
import os
import glob
import argparse
from constants import DATASETS, NO_IN_DIR, CIFAR10, MNIST, OMNIGLOT


here = os.getcwd()
parser = argparse.ArgumentParser(description="""
Convert FoCal (rank-order coded) spike files to 16-bit unsigned integer binary files. Whithin each file a row represents an image:
'rowX = class|Index1|Index2|...|IndexN-1|IndexN' 
""")

parser.add_argument('dataset', type=str, choices=DATASETS, 
                    help='Name of the dataset to convert')
parser.add_argument('input_dir', type=str, default=NO_IN_DIR, 
                    help='Path to the directory containing FoCal files')
parser.add_argument('output_filename', type=str, 
                    default=os.path.join(here, 'output_spikes.bin'),
                    help='Path to the output location of the converted spike files')


def read_file(fname):
    with np.load(fname, allow_pickle=True) as data:

        label = data['label']
        spikes = data['focal_spikes']
        jump = data['grayscale_image'].size
        
        n_spikes = len(spikes)
        max_n_spikes = int(jump * 3 * data['percent'])
        n_spikes = min(n_spikes, max_n_spikes)

        out_data = np.zeros(n_spikes + 1, dtype='uint16')
        out_data[0] = label
        for i, (spike_id, val, cell_type) in enumerate(spikes):
            if(i >= n_spikes):
                break
            out_data[i+1] = cell_type * jump + spike_id
    
        return out_data


def convert_mnist(out_fname, input_dir):
    out_data = None

    idir = os.path.join(input_dir, "**", "**", "*.npz")
    wrong = 0
    correct = 1
    for fname in glob.glob(idir):
        d = read_file(fname)
        

        if out_data is None:
            out_data = d.reshape(1, -1)
        else:
            if d.size < out_data[0].size:
                wrong += 1
                continue

            correct += 1
            out_data = np.vstack([out_data, d])

    # print("num wrong size files {}".format(wrong))
    # print("num correct size files {}".format(correct))
    out_data.tofile(out_fname)


def main():
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        raise Exception(
            'Input directory "{}" not found'.format(args.input_dir))

    if args.dataset.lower() == MNIST.lower():
        cvt = convert_mnist
    # elif args.dataset.lower() == CIFAR10.lower():
    #     import cifar_convert as cvt
    # elif args.dataset.lower() == OMNIGLOT.lower():
    #     import omniglot_convert as cvt
    else:
        raise Exception('Dataset not (yet) supported!')

    cvt(args.output_filename, args.input_dir)
    


if __name__ == '__main__':
    main()





