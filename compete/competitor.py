import argparse
import os
import sys
import cv2
import json


class Competitor(object):
    def __init__(self, kernel_type, config):
        self._kernel_type = kernel_type
        self._conf = config
        print(kernel_type)
        print(config)