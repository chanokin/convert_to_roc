import argparse
import os
import sys
import cv2


def load_config(fname):
    datastore = {}
    with open(fname, 'r') as f:
        datastore = json.load(f)
    return datastore

def parse_dog(data):
    cstd = data['center_std_dev']
    sstd = data['surround_mult'] * cstd
    width = data['kernel_width']
    is_off = data['off_center']
    
    return is_off, width, cstd, sstd

def parse_gauss(data):
    cstd = data['standard_dev']
    width = data['kernel_width']
    
    return width, cstd

def sf_conv(img, conv1d):
    return cv2.sepFilter2D(img, -1, conv1d, conv1d)

class Competitor(object):
    DEF_CONF = os.path.join(os.getcwd(), 'config.json')
    def __init__(self, config_file=DEF_CONF):
        self._conf = load_config(config_file)
        print(self._conf)