
from OpenGL.GL import *
from OpenGL.GLU import *

import math as m
import numpy as np

import cjelly
from drawable import Drawable

class JellyControl( Drawable ) :
	C = 500.0
	K = 10.0

	PTS = np.array( ((0,0,0),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0),(1,0,1),(1,1,1)) , np.float32 )
	EMS = np.array( (0,1, 0,3, 0,5, 1,2, 1,6, 2,3, 2,7, 3,4, 4,5, 4,7, 5,6, 6,7 ) , np.ushort )


	def __init__( self , jelly , pos ) :
		self.pts = jelly.pts
		self.l0  = jelly.l0

		self.frs = np.zeros( (jelly.SHAPE[0],jelly.SHAPE[1],jelly.SHAPE[2],4) , np.float32 )
		self.pl = np.zeros( (jelly.SHAPE[0],jelly.SHAPE[1],jelly.SHAPE[2],4) , np.float32 )
		self.nl = np.zeros( (jelly.SHAPE[0],jelly.SHAPE[1],jelly.SHAPE[2],4) , np.float32 )

		self.set_pos( pos )

	def move( self , d ) :
		self.pos += np.resize( d , 4 )

	def set_pos( self , p ) :
		self.pos = np.resize( np.array(p,np.float32) , 4 )

	def get_pos( self ) :
		return self.pos

	def forces( self , dt , p = None ) :
		if p == None : p = self.pos
		self.nl.fill(0)
		self.frs.fill(0)
		cjelly.control( self.pts , self.nl , p , self.l0 )
		cjelly.update_forces( self.frs , self.nl , self.pl , self.K , self.C , dt )
#        self.pos += np.array( (.0001,0,0) )
		return self.frs

	def draw( self ) :
		glPushMatrix(GL_MODELVIEW)
		glTranslatef( self.pos[0] , self.pos[1] ,self.pos[2] )
		glScalef( self.l0 * 3 , self.l0 * 3 , self.l0 * 3 )
		glTranslatef( -.5 , -.5 , -.5 )
		glColor3f(1,1,0)
		glEnableClientState( GL_VERTEX_ARRAY )
		glVertexPointer( 3 , GL_DOUBLE , 0 , self.PTS )
		glDrawElements( GL_LINES , 24 , GL_UNSIGNED_SHORT , self.EMS )
		glDisableClientState( GL_VERTEX_ARRAY )
		glPopMatrix()

