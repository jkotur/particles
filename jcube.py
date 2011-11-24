
from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

import cjelly
from drawable import Drawable

class JellyCube( Drawable ) :
	SHAPE = (4,4,4)
	C = 1.0
	K = 1.0
	L0 = .1

	def __init__( self ) :
		self.pts = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],3) , np.float64 )
		self.__init_aligned()

	def __init_aligned( self ) :
		for x in range(self.SHAPE[0]) :
			for y in range(self.SHAPE[1]) :
				for z in range(self.SHAPE[2]) :
					self.pts[x,y,z] = np.array((x,y,z))

	def gfx_init( self ) :
		pass

	def draw( self ) :
		glPointSize( 2 )
		glColor3f(0.3,1.0,0)
		glBegin(GL_POINTS)
		for x in range(self.SHAPE[0]) :
			for y in range(self.SHAPE[1]) :
				for z in range(self.SHAPE[2]) :
					glVertex3f( *self.pts[x,y,z] )
		glEnd()


	def wobble( self , dt ) :
		cjelly.wobble( self.pts , dt )

