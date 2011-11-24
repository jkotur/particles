
cimport numpy as np

import cython

import random as rnd
import math as m

import numpy as np

cdef float Gx = 0
cdef float Gy =-9
cdef float Gz = 0

#@cython.boundscheck(False)
cpdef int wobble( np.ndarray[ double , ndim = 4 ] pts , double dt ) :
	return 0

