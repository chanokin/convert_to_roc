import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from focal import Focal
import struct
from constants import *

FOCAL = Focal(mute_output=True)
FOCAL_S = Focal(mute_output=True, small_image=True)
NUM_KERNELS = len(FOCAL.kernels.full_kernels) # four simulated layers

def f2s(f):
    return '{}p{}'.format(int(f), int((f - int(f)) * 1000))

def mkdir(path):
    if sys.version_info[0] < 3:
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        os.makedirs(path, exist_ok=True)

def show(img, cmap=None):
    #shortcut
    plt.matshow(img, cmap=cmap)

def unpickle(file):
    with open(file, 'rb') as fo:
        if sys.version_info[0] < 3:
            d = pickle.load(fo)
        else:
            d = pickle.load(fo, encoding='bytes')
    return d


def ssa_to_img(ssa, scale, img_shape, power_exp=-1.0, start_time=0.0, end_time=np.inf):
    max_places = 4 * np.prod(img_shape)
    max_t = np.max([np.max(ts) for ts in ssa if len(ts)])
    t2p = max_places / max_t
    h, w = img_shape
    n_per_scale = h * w
    start = scale * n_per_scale
    end = start + n_per_scale
    img = np.zeros(img_shape)
    for pix, neuron in enumerate(range(start, end)):
        times = np.asarray(ssa[neuron])
        whr = np.where(np.logical_and(start_time <= times, times < end_time))[0]
        n_spikes = len(whr)
        if n_spikes > 0:
            places = (times[whr] * t2p) + 1
            img[pix//w, pix%w] = np.sum(places**(power_exp))

    
    return img

def num_from_byte_array(fmt, bytes):
    return struct.unpack(fmt, "".join(bytes))[0]
