import argparse
import os
import sys
import cv2
import json
from .utils import load_json
from . import Competitor

CWD = os.path.dirname(os.path.abspath(__file__))

class Competition(object):
    DEF_CONF = os.path.join(CWD, 'config.json')

    def __init__(self, config_file=DEF_CONF):
        self._conf = load_json(config_file)
        self._channels = self._conf['channels']
        self._source_ops = self._create_source_ops(self._conf['sources'])
        self._encoders = self._create_encoders(self._conf['encoders'])
        self._n_encoders = len(self._conf['encoders'])

    def _create_encoders(self, config):
        encs = {}
        for label in config:
            c = config[label]
            kt = c["kernel_type"]
            comps = {k: Competitor(kt, c['competitors'][k]) 
                        for k in c['competitors']}


    def _create_source_ops(self, config):
        """ Assumes RGB order of channels for color images 
        Two actions supported:
        - Select: the operation should return the appropriate channel
        - Convert: use weighted sum to obtain a 'new' channel (e.g. luma, yellow)
        """
        chs = self._channels
        ops = {}
        for label in config:
            act = config[label]['action']
            if act == 'select':
                ch = chs[ config[label]['channel'] ]
                op = lambda x: x[:, :, ch]
                ops[label] = op
            elif act == 'convert':
                ws = {k: config[label][k] if k in config[label] else 0.0 
                        for k in ['red', 'green', 'blue']}
                op = lambda x: x[:, :, chs['red']] * ws['red'] + \
                               x[:, :, chs['green']] * ws['green'] + \
                               x[:, :, chs['blue']] * ws['blue']
                ops[label] = op
        return ops