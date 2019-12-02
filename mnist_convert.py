import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from common import num_from_byte_array
import struct
from focal import focal_to_spike

def read_img_file(filename, start_idx = 0, max_num_images = 10000000000, labels=None):
    f = open(filename, "rb")
    
    try:
        temp = [f.read(1), f.read(1), f.read(1), f.read(1)]
        # magic_number = num_from_byte_array(">I", temp)
        
        temp = [f.read(1), f.read(1), f.read(1), f.read(1)]
        number_images = num_from_byte_array(">I", temp)
        
        temp = [f.read(1), f.read(1), f.read(1), f.read(1)]
        rows_per_image = num_from_byte_array(">I", temp)
        
        temp = [f.read(1), f.read(1), f.read(1), f.read(1)]
        cols_per_image = num_from_byte_array(">I", temp)

        img_idx = 0 
        images = {}
        max_num_images = start_idx + max_num_images
        while img_idx < number_images and img_idx < max_num_images:
        
            if img_idx < start_idx:
                for r in xrange(rows_per_image):
                    for c in xrange(cols_per_image):
                        t = struct.unpack("B", f.read(1))[0]

                img_idx += 1
                continue
            
        img = np.zeros((rows_per_image, cols_per_image))
        for r in xrange(rows_per_image):
            for c in xrange(cols_per_image):
                img[r, c] = struct.unpack("B", f.read(1))[0]
                        
        if labels is not None:
            images[img_idx] = {'img': img, 'lbl': labels[img_idx]}
        else:
            images[img_idx] = {'img': img}

        img_idx += 1
        
    finally:
        f.close()

    return images

def read_label_file(filename, start_idx = 0, max_num_labels = 10000000000000):
    f = open(filename, "rb")
    
    try:
        temp = [f.read(1), f.read(1), f.read(1), f.read(1)]
        # magic_number = num_from_byte_array(">I", temp)
        
        temp = [f.read(1), f.read(1), f.read(1), f.read(1)]
        number_labels = num_from_byte_array(">I", temp)
        
        lbl_idx = start_idx
        max_idx = start_idx + min(number_labels, max_num_labels)
        labels = {}
        while lbl_idx < max_idx:
            labels[lbl_idx] = struct.unpack("B", f.read(1))[0]
            lbl_idx += 1
            
    finally:
        f.close()

    return labels

def mnist_convert(filenames, out_dir, timestep, spikes_per_bin=1, skip_existing=True):
    n_train, n_test = 60000, 10000
    width = height = 28
    batch_size = 1000
    n_imgs = n_train + n_test

    img = np.zeros((width, height))
    spikes = []
    spk_src = []
    
    for start_idx in range(0, n_train, batch_size):

        pc = 100.0*float(start_idx+1)/n_imgs
        sys.stdout.write('\rconverting {:6.2f}%'.format(pc))
        sys.stdout.flush()


        np.savez_compressed(os.path.join(dirname, fname),
            label=label, filename=filename, color_image=img, grayscale_image=gray,
            focal_spikes=spikes, spike_source_array=spk_src, timestep=timestep,
            batch_index=batch_idx, image_batch_index=img_idx)

    print("\tDone with batch!\n")

def open_and_convert(in_dir, out_dir, timestep, spikes_per_bin=1, skip_existing=True):
    search_path = os.path.join(os.getcwd(), in_dir, '*')
    files = sorted( glob.glob(search_path) )
    mnist_convert(files, out_dir, timestep, spikes_per_bin, skip_existing)

