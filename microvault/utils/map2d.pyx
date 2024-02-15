# distutils: language=c++

from libcpp cimport bool
from libcpp.pair cimport pair as cpp_pair
from libcpp.queue cimport priority_queue as cpp_priority_queue

import numpy as np

cimport cython
cimport numpy as np
from cython.operator cimport dereference as deref

from math import sqrt

from libc.math cimport acos as cacos
from libc.math cimport cos as ccos
from libc.math cimport floor as cfloor
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt

import os

from matplotlib.pyplot import imread
from yaml import SafeLoader, load


cdef class CMap2D:

    # define variables
    cdef public np.float32_t[:,::1] _occupancy 
    cdef int occupancy_shape0
    cdef int occupancy_shape1
    cdef float resolution_ 
    cdef float _thresh_occupied
    cdef float thresh_free
    cdef float HUGE_ 
    cdef public np.float32_t[:] origin  

    # class init
    def __init__(self, folder=None, name=None, silent=False):
        if folder is None or name is None:
            return
            
        folder = os.path.expanduser(folder)
        yaml_file = os.path.join(folder, name + ".yaml")

        if not silent:
            print("Loading map definition from {}".format(yaml_file))

        with open(yaml_file) as stream:
            mapparams = load(stream, Loader=SafeLoader)
        map_file = os.path.join(folder, mapparams["image"])

        if not silent:
            print("Map definition found. Loading map from {}".format(map_file))

        mapimage = imread(map_file)
        temp = (1. - mapimage.T[:, ::-1] / 254.).astype(np.float32)
        mapimage = np.ascontiguousarray(temp)
        self._occupancy = mapimage
        self.occupancy_shape0 = mapimage.shape[0]
        self.occupancy_shape1 = mapimage.shape[1]
        self.resolution_ = mapparams["resolution"]
        self.origin = np.array(mapparams["origin"][:2]).astype(np.float32)

        if mapparams["origin"][2] != 0:
            raise ValueError("Map origin z coordinate must be 0")

        self._thresh_occupied = mapparams["occupied_thresh"]
        self.thresh_free = mapparams["free_thresh"]

        print("Thresh occupied: {}. Thresh free: {}".format(self._thresh_occupied, self.thresh_free))
        print("Map loaded. Origin: {}. Resolution: {}. Occupancy shape: {}".format(np.array(self.origin), self.resolution_, self._occupancy.shape))
        self.HUGE_ = 100 * self.occupancy_shape0 * self.occupancy_shape1
        
        if self.resolution_ == 0:
            raise ValueError("resolution can not be 0")