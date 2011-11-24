
from OpenGL.GL import *
from OpenGL.GLU import *

import transformations as tr

from drawable import Drawable

class Plane( Drawable ) :
	def __init__( self , size , m ) :
		Drawable.__init__( self )

		self.size = map( lambda x : x*.5 , size )
		self.m = m

	def draw( self ) :
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glMultTransposeMatrixf(self.m)
		self._quad()
		glPopMatrix()

	def _quad( self ) :
		glBegin( GL_QUADS )
		glNormal3f(0,1,0)
		glVertex3f(-self.size[0],0,-self.size[1])
		glVertex3f( self.size[0],0,-self.size[1])
		glVertex3f( self.size[0],0, self.size[1])
		glVertex3f(-self.size[0],0, self.size[1])
		glEnd()

