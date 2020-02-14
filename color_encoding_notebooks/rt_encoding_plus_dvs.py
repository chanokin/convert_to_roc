%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
# from focal import Focal, focal_to_spike, spike_trains_to_images_g
from scipy.signal import convolve2d
from scipy import misc
import cv2

def slice2str(s, e):
    return "{}: {}".format(s, e)

def _to_str(rs, re, cs, ce):
    return "{}, {}".format(slice2str(rs, re), slice2str(cs, ce))

def convolve(img, conv1d):
    return cv2.sepFilter2D(img, -1, conv1d, conv1d)

class pixel(object):
    def __init__(self, row, col, val, image_idx):
        self.r = row
        self.c = col
        self.v = val
        self.x = image_idx

class LinkedPixel():
    def __init__(self, mtx, val, row, col, 
                 east=None, north=None, west=None, south=None, 
                 higher=None, lower=None):
        self._matrix = mtx
        self._val = val
        self._row = row 
        self._col = col
        self._east = east
        self._nort = north
        self._west = west
        self._south = south
        self._higher = higher
        self._lower = lower
    
    def __eq__(self, other):
        return (self.v == other.v and \
                self.r == other.r and \
                self.c == other.c)
    
    @property
    def mat_id(self):
        return self._matrix._id
    
    @property
    def n(self):
        return self._north

    @n.setter
    def n(self, v):
        self._north = v

    @property
    def w(self):
        return self._west

    @w.setter
    def w(self, v):
        self._west = v
    
    @property
    def s(self):
        return self._south
    
    @s.setter
    def s(self, v):
        self._south = v

    @property
    def e(self):
        return self._east

    @e.setter
    def e(self, v):
        self._east = v

    @property
    def hi(self):
        return self._higher

    @hi.setter
    def hi(self, v):
        self._higher = v
    
    @property
    def lo(self):
        return self._lower

    @lo.setter
    def lo(self, v):
        self._lower = v
    
    @property
    def v(self):
        return self._val

    @v.setter
    def v(self, val):
        self._val = val

    
class SortedLinkedList():
    def __init__(self, pixel_list=[]):
        self._start = None
        self._end = None
        self._n_links = max(0, int(np.round(np.log(len(pixel_list)))))
        self._mid_links = []
        self._size = 0
    
        
    def insert(self, pix):
        def _in(lo, mid, hi):
            if lo is not None:
                lo.hi = mid
            mid.lo = lo
            mid.hi = hi
            if hi is not None:
                hi.lo = mid
        
        v = pix.v
        if self._start is None:
            self._start = pix
            self._end = pix
            self._start.hi = self._end
            self._end.lo = self._start
            self._size += 1
                
        
        elif v <= self._start.v:
            tmp = self._start
            self._start = pix
            _in(None, pix, tmp)
            self._size += 1
            return
        
        elif v >= self._end.v:
            tmp = self._end
            self._end = pix
            _in(tmp, pix, None)
            self._size += 1
            return

        
        elif len(self._mid_links):
            ml = self._mid_links[-1]
            inserted = False
            if v > ml.v:
                hi = ml.hi
                while hi is not None:
                    if v <= hi.v:
                        tmp = hi.lo
                        _in(tmp, pix, hi)
                        self._size += 1
                        inserted = True
                    hi = hi.hi

        
            ml = self._mid_links[0]
            if v <= ml.v and not inserted:
                lo = ml.lo
                while hi is not None:
                    if v > lo.v:
                        tmp = lo.hi
                        _in(lo, pix, tmp)
                        self._size += 1
                    lo = lo.lo
        
        else:
            for i, ml in enumerate(self._mid_links[1:]):
                lo = self._mid_links[i-1]
                if v > lo.v and v <= ml.v:
                    n = ml
                    while n != lo:
                        tmp = n.lo
                        if tmp.v > v and v <= n.v:
                            _in(tmp, pix, n)
                            self._size += 1

                        n = n.lo
        
        self.__redo_mids(self)
    
    def __redo_mids(self):
        ls = np.floor(np.log(self._size))
        if ls == 0:
            return
        if ls > len(self._mid_links):
            self._mid_links.append(self._end)

            
    
    def __str__(self):
        vals = []
        n = self._start
        while n is not None:
            vals.append(n.v)
            n = n.hi
        return "[{}]".format(", ".join(vals))


class LinkedImage():
    def __init__(self, width, height, idx=None):
        self._id = idx if idx is not None else np.random.randint(0, 10000)
        self.width = width
        self.height = height

        self._pixels = {r: {c:  LinkedPixel(self, None, r, c)
                        for c in range(width)} for r in range(height)}
        self.__link()
        
    def __init__(self, img, idx=None):
        if not isinstance(img, np.ndarray):
            raise Exception("LinkedImage: initialization from a non ND-array is not available")

        height, width = img.shape
        self.width = width
        self.height = height

        self._id = idx if idx is not None else np.random.randint(0, 10000)
        self._pixels = {r: {c:  LinkedPixel(self, img[r, c], r, c)
                        for c in range(width)} for r in range(height)}
        self.__link()
        
    def __link(self):
        def _l(p, n, w, s, e):
            p.n = n
            p.w = w
            p.s = s
            p.e = e

        for r in range(1, self.height):
            for c in range(1, self.width):
                _l(self._pixels[r][c],
                   self._pixels[r-1][c],
                   self._pixels[r][c-1],
                   self._pixels[r+1][c],
                   self._pixels[r][c+1])
