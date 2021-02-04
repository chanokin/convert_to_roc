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
        height, width = data['grayscale_image'].shape
        keepers = data['keepers'].item()
        sizes = {
            k: (height // keepers[k][0]) * (width // keepers[k][1])
            for k in keepers
        }
        # total_neurons = np.sum([sizes[k] for k in sizes])
        jumps = [0]
        for k in sorted(sizes.keys()):
            if k == 0:
                continue
            jumps.append(sizes[k - 1] + jumps[k-1])

        n_spikes = len(spikes)
        max_n_spikes = int(jump * 4 * data['percent'])
        n_spikes = min(n_spikes, max_n_spikes)

        out_data = np.zeros(n_spikes, dtype='uint16')
        # out_data[0] = label
        for i, (spike_id, val, cell_type) in enumerate(spikes):
            if(i >= n_spikes):
                break

            row, col = spike_id // width, spike_id % width
            row, col = (row // keepers[cell_type][1],
                        col // keepers[cell_type][0])

            new_width = width // keepers[cell_type][1]
            spike_id = row * new_width + col

            out_data[i] = jumps[int(cell_type)] + spike_id
    
        return out_data, jump * 4


def convert_mnist(out_fname, input_dir):
    def key_f(entry):
        d = entry[-10:-4]
        #print(entry, d)
        return int(d)

    out_data = None

    idir = os.path.join(input_dir, "**", "**", "*.npz")
    wrong = 0
    correct = 1
    fnames = glob.glob(idir)

    training = [x for x in fnames if 'train' in x]
    training = sorted(training, key=key_f)

    testing = [x for x in fnames if 't10k' in x]
    testing = sorted(testing, key=key_f)

    sizes = [read_file(fname)[0].size for fname in fnames]
    ss = int(np.min(sizes))
    total = float(len(sizes))
    tinv = 1./total
    for i, fname in enumerate(fnames):
        sys.stdout.write("\rConverted {:6.2f}%".format( 100.0*(i+1.0)*tinv ))
        sys.stdout.flush()

        d, s = read_file(fname)

        if d.size < ss:
            wrong += 1
            continue

        if out_data is None:
            out_data = d[:ss].reshape(1, -1)
        else:
            out_data = np.vstack([out_data, d[:ss]])

        correct += 1

    # print("num wrong size files {}".format(wrong))
    # print("num correct size files {}".format(correct))
    out_fname = "{0}_n_spikes_per_sample_{1}_total_neurons_{2}{3}".format(
        out_fname[:-4], out_data[0].size, s, out_fname[-4:]
    )

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





