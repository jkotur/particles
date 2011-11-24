
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
	if dl == 0 : return np.zeros( 3 , np.float64 )
	cdef double l = l0 - dl
#    print l0 , dl , l , dp / dl * l
	return dp / dl * l

cpdef int control( np.ndarray[ double , ndim = 4 ] pts ,
                   np.ndarray[ double , ndim = 4 ] nl  , 
				   np.ndarray[ double , ndim = 1 ] ctl ,
				   double l ) :
	cdef np.ndarray[ double ] p = np.zeros(3)
	cdef int x = pts.shape[0] - 1
	cdef int y = pts.shape[1] - 1
	cdef int z = pts.shape[2] - 1
	l = l * 3.0 / 2.0
	p[0] = ctl[0] + l
	p[1] = ctl[1] + l
	p[2] = ctl[2] + l
	nl[x,y,z] += spring_force( pts[x,y,z], p , 0 )
	p[0] = ctl[0] + l
	p[1] = ctl[1] + l
	p[2] = ctl[2] - l
	nl[x,y,0] += spring_force( pts[x,y,0], p , 0 )
	p[0] = ctl[0] + l
	p[1] = ctl[1] - l
	p[2] = ctl[2] - l
	nl[x,0,0] += spring_force( pts[x,0,0], p , 0 )
	p[0] = ctl[0] - l
	p[1] = ctl[1] - l
	p[2] = ctl[2] - l
	nl[0,0,0] += spring_force( pts[0,0,0], p , 0 )
	p[0] = ctl[0] - l
	p[1] = ctl[1] - l
	p[2] = ctl[2] + l
	nl[0,0,z] += spring_force( pts[0,0,z], p , 0 )
	p[0] = ctl[0] - l
	p[1] = ctl[1] + l
	p[2] = ctl[2] + l
	nl[0,y,z] += spring_force( pts[0,y,z], p , 0 )
	p[0] = ctl[0] - l
	p[1] = ctl[1] + l
	p[2] = ctl[2] - l
	nl[0,y,0] += spring_force( pts[0,y,0], p , 0 )
	p[0] = ctl[0] + l
	p[1] = ctl[1] - l
	p[2] = ctl[2] + l
	nl[x,0,z] += spring_force( pts[x,0,z], p , 0 )
	return 0

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
	return 0

cpdef int update( np.ndarray[ double , ndim = 4 ] pts ,
                  np.ndarray[ double , ndim = 4 ] prv ,
                  np.ndarray[ double , ndim = 4 ] frs ,
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
				f = frs[x,y,z]
				f[0] += Gx
				f[1] += Gy
				f[2] += Gz
				a = f / mas[x,y,z]
				n = a * dt * dt + 2 * pts[x,y,z] - prv[x,y,z]
				prv[x,y,z] = pts[x,y,z]
				pts[x,y,z] = n
				z += 1
			y += 1
		x += 1

cpdef int update_forces( np.ndarray[ double , ndim = 4 ] frs ,
                         np.ndarray[ double , ndim = 4 ] nl  ,
                         np.ndarray[ double , ndim = 4 ] pl  ,
				         double k , double c ,
                         double dt ) :
	cdef int x , y , z
	x = 0
	while x < frs.shape[0] :
		y = 0
		while y < frs.shape[1] :
			z = 0
			while z < frs.shape[2] :
				lp = ( nl[x,y,z] - pl[x,y,z] ) / ( 2 * dt )
				frs[x,y,z] += -k * lp - c * nl[x,y,z]
				pl[x,y,z] = nl[x,y,z]
				z += 1
			y += 1
		x += 1

cpdef int collisions( np.ndarray[ double , ndim = 4 ] pts ,
                      np.ndarray[ double , ndim = 4 ] prv ,
					  np.ndarray[ double , ndim = 1 ] brd ,
					  double u ) :
	cdef int x , y , z , b , i
	x = 0
	while x < pts.shape[0] :
		y = 0
		while y < pts.shape[1] :
			z = 0
			while z < pts.shape[2] :
				i = 0 
				while i < 3 :
					b = 0
					# maximum number of collisions with border in one frame
					while b < 3 : 
						if pts[x,y,z,b] <= brd[2*b] or \
						   pts[x,y,z,b] >= brd[2*b+1] :
							n = prv[x,y,z,b]+u*pts[x,y,z,b]-u*prv[x,y,z,b]
							pts[x,y,z,b] = prv[x,y,z,b]
							prv[x,y,z,b] = n
						b += 1
					i += 1
				z += 1
			y += 1
		x += 1

