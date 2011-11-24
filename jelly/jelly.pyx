
cimport numpy as np

import cython

import random as rnd
import math as m

import numpy as np

cdef float Gx = 0
cdef float Gy =-9
cdef float Gz = 0

#@cython.boundscheck(False)
cpdef int wobble( np.ndarray[ double , ndim = 4 ] pts ,
                  np.ndarray[ double , ndim = 4 ] prv ,
                  np.ndarray[ double , ndim = 4 ] frs ,
                  np.ndarray[ double , ndim = 4 ] mas ,
                  double dt ) :
	springs( pts , frs )
	update ( pts , prv , frs , mas , dt )

cpdef int springs( np.ndarray[ double , ndim = 4 ] pts ,
                   np.ndarray[ double , ndim = 4 ] frs ) :
	cdef int x , y , z

	x = 0
	while x < pts.shape[0] :
		y = 0
		while y < pts.shape[1] :
			z = 0
			while z < pts.shape[2] :
				frs[x,y,z,0] = Gx
				frs[x,y,z,1] = Gy 
				frs[x,y,z,2] = Gz
				z += 1
			y += 1
		x += 1


cpdef int update( np.ndarray[ double , ndim = 4 ] pts ,
                  np.ndarray[ double , ndim = 4 ] prv ,
                  np.ndarray[ double , ndim = 4 ] frs ,
                  np.ndarray[ double , ndim = 4 ] mas ,
                  double dt ) :
	cdef np.ndarray a = np.zeros( 3 )
	cdef np.ndarray v = np.zeros( 3 )
	cdef np.ndarray n = np.zeros( 3 )

	cdef int x , y , z
	x = 0
	while x < pts.shape[0] :
		y = 0
		while y < pts.shape[1] :
			z = 0
			while z < pts.shape[2] :
				a = frs[x,y,z] / mas[x,y,z]
				n = a * dt * dt + 2 * pts[x,y,z] - prv[x,y,z]
				v  = ( n - prv[x,y,z] ) / ( 2 * dt )
				prv[x,y,z] = pts[x,y,z]
				pts[x,y,z] = n
				z += 1
			y += 1
		x += 1

