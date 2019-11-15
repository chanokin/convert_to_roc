import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from focal import *


fcl = Focal()
num_kernels = len(fcl.kernels.full_kernels) # four simulated layers
tstep = 1.0

def f2s(f):
    return '{}p{}'.format(int(f), int((f - int(f)) * 1000))

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def show(img, cmap=None):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show()


def extract_row(img, width, height, channels):
    return np.rollaxis(img.reshape(channels, width, height), 0, channels)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)#, encoding='bytes')
    return dict

def open_and_convert(data):
    here = os.getcwd()
    n_imgs, n_pixels = data['data'].shape
    channels = 3 #rgb images!
    width = height = int(np.sqrt(n_pixels/channels))
    img = np.zeros((width, height, channels))
    gray = np.zeros((width, height))
    spikes = []
    spk_src = []
    batch_idx = data['batch_label'][-6]
    for img_idx in range(n_imgs)[:]:
        sys.stdout.write('\rconverting {:6.2f}%'.format(100.0*float(img_idx+1)/n_imgs))
        sys.stdout.flush()

        img[:] = extract_row(data['data'][img_idx], width, height, channels)
        gray[:] = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY).astype('float')
        label = data['labels'][img_idx]
        filename = data['filenames'][img_idx]

        dirname = os.path.join(here, 'spike_files')
        makedir(dirname)

        dirname = os.path.join(dirname, 'class_{:06d}'.format(label))
        makedir(dirname)

        spikes[:] = fcl.apply(gray)
        spk_src[:] = focal_to_spike(spikes, gray.shape, 
                                    spikes_per_time_block=1, 
                                    start_time=0., time_step=1.)

        fname = 'class_{:06d}_timestep_{}_batch_{}_index_{:06d}.npz'.\
                    format(label, f2s(tstep), batch_idx, img_idx)
        np.savez_compressed(os.path.join(dirname, fname),
            label=label, filename=filename, color_image=img, grayscale_image=gray,
            focal_spikes=spikes, spike_source_array=spk_src, timestep=tstep,
            batch_index=batch_idx, image_batch_index=img_idx)

    print("Done with batch!\n")

def main():
    in_dir = 'cifar-10-batches-py'
    search_path = os.path.join(os.getcwd(), in_dir, '*')
    files = sorted( glob.glob(search_path) )
    for f in files:
        if '_batch' in f:
            print(f)
            open_and_convert(unpickle(f))


if __name__ == '__main__':
    main()
