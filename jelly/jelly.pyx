
cimport numpy as np

import cython

import random as rnd
import math as m

import numpy as np


cdef float Gx = 0
cdef float Gy =-9
cdef float Gz = 0

#@cython.boundscheck(False)
cpdef np.ndarray[ double ] spring_force( np.ndarray[ double ] p0 ,
                        np.ndarray[ double ] p1 ,
						double l0 ) :
	cdef np.ndarray[ double ] dp = p1 - p0
	cdef double dl = np.linalg.norm( dp )
	cdef double l = l0 - dl
	return dp / dl * l

cpdef int springs( np.ndarray[ double , ndim = 4 ] pts ,
                   np.ndarray[ double , ndim = 4 ] nl , 
				   double l0 , double l1 ) :
	cdef int x , y , z

	x = 0
	while x < pts.shape[0] :
		y = 0
		while y < pts.shape[1] :
			z = 0
			while z < pts.shape[2] :
				nl[x,y,z,0] = 0
				nl[x,y,z,1] = 0
				nl[x,y,z,2] = 0

				if x+1 < pts.shape[0] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x+1,y,z],l0)
				if y+1 < pts.shape[1] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x,y+1,z],l0)
				if z+1 < pts.shape[2] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x,y,z+1],l0)
				if x-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x-1,y,z],l0)
				if y-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x,y-1,z],l0)
				if z-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x,y,z-1],l0)
				if x+1 < pts.shape[0] and y+1 < pts.shape[1] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x+1,y+1,z],l1)
				if y+1 < pts.shape[1] and z+1 < pts.shape[2] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x,y+1,z+1],l1)
				if z+1 < pts.shape[2] and x+1 < pts.shape[0] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x+1,y,z+1],l1)
				if x-1 >= 0 and y+1 < pts.shape[1] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x-1,y+1,z],l1)
				if y-1 >= 0 and z+1 < pts.shape[2] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x,y-1,z+1],l1)
				if z-1 >= 0 and x+1 < pts.shape[0] :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x+1,y,z-1],l1)
				if x+1 < pts.shape[0] and y-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x+1,y-1,z],l1)
				if y+1 < pts.shape[1] and z-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x,y+1,z-1],l1)
				if z+1 < pts.shape[2] and x-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x-1,y,z+1],l1)
				if x-1 >= 0 and y-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x-1,y-1,z],l1)
				if y-1 >= 0 and z-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x,y-1,z-1],l1)
				if z-1 >= 0 and x-1 >= 0 :
					nl[x,y,z]+=spring_force(pts[x,y,z],pts[x-1,y,z-1],l1)
				z += 1
			y += 1
		x += 1


cpdef int update( np.ndarray[ double , ndim = 4 ] pts ,
                  np.ndarray[ double , ndim = 4 ] prv ,
                  np.ndarray[ double , ndim = 4 ] pl  ,
                  np.ndarray[ double , ndim = 4 ] nl  ,
                  np.ndarray[ double , ndim = 4 ] mas ,
				  double k , double c ,
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
				dl  = ( nl[x,y,z] - pl[x,y,z] ) / ( 2 * dt )
				f = -k * dl - c * nl[x,y,z]
				if y == 3 and x >= 1 and x <=2 and z >= 1 and z <= 2 :
					f[1] -= 9
				a = f / mas[x,y,z]
				n = a * dt * dt + 2 * pts[x,y,z] - prv[x,y,z]
				prv[x,y,z] = pts[x,y,z]
				pts[x,y,z] = n
				z += 1
			y += 1
		x += 1

