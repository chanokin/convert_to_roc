import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from common import *
from focal import focal_to_spike

def row_to_mat(img, width, height, channels):
    return np.rollaxis(img.reshape(channels, width, height), 0, channels)

def cifar_convert(data, out_dir, timestep, spikes_per_bin=1, skip_existing=True):
    n_imgs, n_pixels = data['data'].shape
    channels = 3 #rgb images!
    width = height = int(np.sqrt(n_pixels/channels)) #assuming squared images
    img = np.zeros((width, height, channels))
    gray = np.zeros((width, height))
    spikes = []
    spk_src = []
    batch_idx = 'test' if 'test' in data['batch_label'] else data['batch_label'][-6]
    
    for img_idx in range(n_imgs)[:]:
        pc = 100.0*float(img_idx+1)/n_imgs
        sys.stdout.write('\rconverting {:6.2f}%'.format(pc))
        sys.stdout.flush()

        label = data['labels'][img_idx]
        fname = 'class_{:06d}_timestep_{}_batch_{}_index_{:06d}.npz'.\
                    format(label, f2s(timestep), batch_idx, img_idx)
        mkdir(out_dir)

        dirname = os.path.join(out_dir, 'class_{:06d}'.format(label))
        mkdir(dirname)

        if skip_existing and os.path.exists(os.path.join(dirname, fname)):
            continue

        img[:] = row_to_mat(data['data'][img_idx], width, height, channels)
        gray[:] = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY).astype('float')
        filename = data['filenames'][img_idx]


        spikes[:] = FOCAL.apply(gray)
        spk_src[:] = focal_to_spike(spikes, gray.shape, 
                                    spikes_per_time_block=spikes_per_bin, 
                                    start_time=0., time_step=timestep)

        np.savez_compressed(os.path.join(dirname, fname),
            label=label, filename=filename, color_image=img, grayscale_image=gray,
            focal_spikes=spikes, spike_source_array=spk_src, timestep=timestep,
            batch_index=batch_idx, image_batch_index=img_idx)

    print("\tDone with batch!\n")

def open_and_convert(in_dir, out_dir, timestep, spikes_per_bin=1, skip_existing=True):
    search_path = os.path.join(os.getcwd(), in_dir, '*')
    files = sorted( glob.glob(search_path) )
    for f in files:
        # Note: assuming the standard naming convention still holds a.k.a.
        #       the user just extracted to a directory
        if '_batch' in f: 
            print("Converting batch: {}".format(f))
            cifar_convert(unpickle(f), out_dir, timestep, spikes_per_bin, skip_existing)


