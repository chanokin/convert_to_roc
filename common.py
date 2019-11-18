import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from focal import Focal


FOCAL = Focal(mute_output=True)
NUM_KERNELS = len(FOCAL.kernels.full_kernels) # four simulated layers
NO_IN_DIR = 'DID_NOT_PASS_THE_INPUT_DIR'
CIFAR10, MNIST, OMNIGLOT = 'cifar10', 'mnist', 'omniglot'
DATASETS = [CIFAR10, MNIST, OMNIGLOT,]

def f2s(f):
    return '{}p{}'.format(int(f), int((f - int(f)) * 1000))

def mkdir(path):
    if sys.version_info[0] < 3:
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        os.makdedirs(path, exist_ok=True)

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
