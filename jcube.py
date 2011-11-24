
from OpenGL.GL import *
from OpenGL.GLU import *

import math as m
import numpy as np
import random as rnd

import cjelly
from drawable import Drawable

class JellyCube( Drawable ) :
	SHAPE = (4,4,4)
	C = 5.0
	K = 1.0

	def __init__( self ) :
		self.pts = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],3) , np.float64 )
		self.prv = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],3) , np.float64 )
		self.pl  = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],3) , np.float64 )
		self.nl  = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],3) , np.float64 )
		self.frs = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],3) , np.float64 )
		self.mas = np.ones ( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],3) , np.float64 )
		self.l0 = 1
		self.l1 = m.sqrt(2) * self.l0

		self.__func( self.pts , lambda x,y,z : np.array((x,y,z)) * self.l0 )
		self.__func( self.prv , lambda x,y,z : self.pts[x,y,z] )
		self.__func( self.pl  , lambda x,y,z : 0 )
		self.__func( self.mas , lambda x,y,z : 1 )
		self.pts[2,3,2,1] += 2.
		self.prv[2,3,2,1] += 2
		self.pts[1,3,1,1] -= 2
		self.prv[1,3,1,1] -= 2

		self.ctlpos = None

	def __func( self , p , f ) :
		for x in range(self.SHAPE[0]) :
			for y in range(self.SHAPE[1]) :
				for z in range(self.SHAPE[2]) :
					p[x,y,z] = f( x,y,z )

	def gfx_init( self ) :
		pass

	def draw( self ) :
		glPointSize( 5 )
		glBegin(GL_POINTS)
		for x in range(self.SHAPE[0]) :
			for y in range(self.SHAPE[1]) :
				for z in range(self.SHAPE[2]) :
					glColor3f(x/4.0 , y/4.0 , z/4.0 )
					glVertex3f( *self.pts[x,y,z] )
		glEnd()


	def wobble( self , dt , addfrs ) :
		self.nl.fill(0)
		self.frs.fill(0)

		cjelly.springs( self.pts , self.nl , self.l0 , self.l1 )

		cjelly.update_forces( self.frs , self.nl , self.pl , self.K , self.C , dt )
		self.frs += addfrs

		cjelly.update( self.pts , self.prv , self.frs , self.mas , self.K ,self.C , dt )

