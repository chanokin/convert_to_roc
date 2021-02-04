import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from common import num_from_byte_array, mkdir, f2s, FOCAL_S
import struct
from focal import focal_to_spike
try:
    x = xrange(1)
except:
    xrange = range


def read_img_file(filename, start_idx = 0, max_num_images = 10000000000, labels=None):
    f = open(filename, "rb")
    
    try:
        temp = [f.read(1), f.read(1), f.read(1), f.read(1)]
        # magic_number = num_from_byte_array(">I", temp)
        
        temp = [f.read(1) for _ in range(4)]
        tmp = temp[0]
        for i in range(1, len(temp)):
            tmp += temp[i]
        number_images = num_from_byte_array(">I", tmp)
        # print("Number of Images {}".format(number_images))
        
        temp = [f.read(1) for _ in range(4)]
        tmp = temp[0]
        for i in range(1, len(temp)):
            tmp += temp[i]
        rows_per_image = num_from_byte_array(">I", tmp)
        
        temp = [f.read(1) for _ in range(4)]
        tmp = temp[0]
        for i in range(1, len(temp)):
            tmp += temp[i]
        cols_per_image = num_from_byte_array(">I", tmp)
        # print("Image resolution {}x{}".format(rows_per_image, cols_per_image))
        
        ### 1 <=> from current position
        f.seek(start_idx * rows_per_image * cols_per_image, 1) 

        img_idx = start_idx
        images = {}
        max_num_images = min(start_idx + max_num_images, number_images)
        while img_idx < max_num_images:
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
        temp = [f.read(1) for _ in range(4)]
        # magic_number = num_from_byte_array(">I", temp)
        
        temp = [f.read(1) for _ in range(4)]
        tmp = temp[0]
        for i in range(1, len(temp)):
            tmp += temp[i]

        number_labels = num_from_byte_array(">I", tmp)
        # print("Number of Labels = {}".format(number_labels))
        
        # 1 <==> from current position
        f.seek(start_idx, 1) 

        lbl_idx = start_idx
        max_idx = min(number_labels, start_idx + max_num_labels)
        labels = {}
        while lbl_idx < max_idx:
            labels[lbl_idx] = struct.unpack("B", f.read(1))[0]
            lbl_idx += 1
            
    finally:
        f.close()

    return labels


def process(labels, images, n_imgs, out_dir, spikes_per_bin, timestep, 
            log_offset=0, percent=0.3):

    if len(images) == 0:
        print('no images found!')
        return
    keepers = {ct: FOCAL_S.convolver.get_subsample_keepers(ct)
               for ct in range(FOCAL_S.kernels.max_num_kernels)}
    firstkey = list(images.keys())[0]
    img = np.zeros_like(images[firstkey]['img'])
    spikes = []
    spk_src = []
    for img_idx in labels:
        pc = 100.0*float(img_idx + 1 + log_offset)/n_imgs
        sys.stdout.write('\rconverting {:6.2f}%'.format(pc))
        sys.stdout.flush()

        label = labels[img_idx]
        img[:] = images[img_idx]['img']

        spikes[:] = FOCAL_S.apply(img, percent)
        spk_src[:] = focal_to_spike(spikes, img.shape,
                                    spikes_per_time_block=spikes_per_bin,
                                    start_time=0., time_step=timestep)

        dirname = os.path.join(out_dir, "{:d}".format(label))
        mkdir(dirname)
        fname = 'class_{:06d}_timestep_{}_index_{:06d}.npz'.\
            format(label, f2s(timestep), img_idx)

        np.savez_compressed(os.path.join(dirname, fname),
            label=label, color_image=img, grayscale_image=img,
            focal_spikes=spikes, spike_source_array=spk_src, timestep=timestep,
            image_batch_index=img_idx, kernels=FOCAL_S.kernels.full_kernels,
            percent=percent, keepers=keepers)

def mnist_convert(filenames, out_dir, percent, timestep, spikes_per_bin=1, 
                  skip_existing=True):
    n_train, n_test = 60000, 10000
    batch_size = 1000
    n_imgs = n_train + n_test

    labels = {}
    images = {}

    print("Converting training set")
    train_dir = os.path.join(out_dir, 'train')
    mkdir(train_dir)
    img_fname = [f for f in filenames if 'train-images' in f][0]
    lbl_fname = [f for f in filenames if 'train-labels' in f][0]
    for start_idx in range(0, n_train, batch_size):
        labels.clear()
        labels = read_label_file(lbl_fname, start_idx, batch_size)
        images.clear()
        images = read_img_file(img_fname, start_idx, batch_size, labels)

        process(labels, images, n_train, train_dir, spikes_per_bin,
                timestep, 0, percent)

    print("\n\nConverting test set")
    test_dir = os.path.join(out_dir, 't10k')
    mkdir(test_dir)
    img_fname = [f for f in filenames if 't10k-images' in f][0]
    lbl_fname = [f for f in filenames if 't10k-labels' in f][0]
    for start_idx in range(0, n_test, batch_size):
        labels.clear()
        labels = read_label_file(lbl_fname, start_idx, batch_size)
        images.clear()
        images = read_img_file(img_fname, start_idx, batch_size, labels)

        process(labels, images, n_test, test_dir, spikes_per_bin,
                timestep, 0, percent)


def open_and_convert(in_dir, out_dir, percent, timestep, 
                     spikes_per_bin=1, skip_existing=True, scaling=1.0):
    search_path = os.path.join(in_dir, '*')
    files = sorted( glob.glob(search_path) )
    mnist_convert(files, out_dir, percent, timestep, spikes_per_bin, skip_existing)

