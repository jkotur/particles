
from OpenGL.GL import *
from OpenGL.GLU import *

import math as m
import numpy as np

from mesh import Mesh

class Cube( Mesh ) :
	def __init__( self , n ) :
		Mesh.__init__( self , buffers = self.gen_v( n , n ) )

	def gen_v( self , nx , ny ) :
		nx += 1
		ny += 1

		v = np.zeros( (6,nx,ny,3)   , np.float64 )
		n = np.zeros( (6,nx,ny,3)   , np.float64 )
		t = np.zeros( (6,2,nx-1,ny-1,3) , np.uint32  )

		for x in range(nx) :
			for y in range(ny) :
				v[0,x,y] = np.array(( 0 , x/float(nx-1) , y/float(ny-1) ))
				v[1,x,y] = np.array(( 1 , x/float(nx-1) , y/float(ny-1) ))
				v[2,x,y] = np.array(( x/float(nx-1) , 1 , y/float(ny-1) ))
				v[3,x,y] = np.array(( x/float(nx-1) , 0 , y/float(ny-1) ))
				v[4,x,y] = np.array(( x/float(nx-1) , y/float(ny-1) , 0 ))
				v[5,x,y] = np.array(( x/float(nx-1) , y/float(ny-1) , 1 ))

				n[0,x,y] = np.array((-1 , 0 , 0 ))
				n[1,x,y] = np.array(( 1 , 0 , 0 ))
				n[2,x,y] = np.array(( 0 , 1 , 0 ))
				n[3,x,y] = np.array(( 0 ,-1 , 0 ))
				n[4,x,y] = np.array(( 0 , 0 ,-1 ))
				n[5,x,y] = np.array(( 0 , 0 , 1 ))

		for y in range(ny-1) :
			for x in range(nx-1) :
				for i in range(0,6,2) :
					t[i,0,x,y] = np.array(( 0, 1, nx))+ x + y*nx + i*nx*ny
					t[i,1,x,y] = np.array((1,nx+1,nx))+ x + y*nx + i*nx*ny
				for i in range(1,6,2) :
					t[i,0,x,y] = np.array(( 0, nx, 1))+ x + y*nx + i*nx*ny
					t[i,1,x,y] = np.array((1,nx,nx+1))+ x + y*nx + i*nx*ny

		return v , n , t

