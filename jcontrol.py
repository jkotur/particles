
from OpenGL.GL import *
from OpenGL.GLU import *

import pycuda.autoinit
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda.gpuarray as cuda_garr

import math as m
import numpy as np

import cjelly
from drawable import Drawable

class JellyControl( Drawable ) :
	C = 5000.0
	K = 100.0

	PTS = np.array( ((0,0,0),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0),(1,0,1),(1,1,1)) , np.float32 )
	EMS = np.array( (0,1, 0,3, 0,5, 1,2, 1,6, 2,3, 2,7, 3,4, 4,5, 4,7, 5,6, 6,7 ) , np.ushort )


	def __init__( self , jelly , pos ) :
		self.jelly = jelly
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

	def gfx_init( self ) :
		self.gpts = self.jelly.gpts

		self.grid = (1,1)
		self.block = (4,4,4)

		self.dfrs = cuda_driver.mem_alloc( self.frs.nbytes )
		self.dnl  = cuda_driver.mem_alloc( self.nl .nbytes )
		self.dpl  = cuda_driver.mem_alloc( self.pl .nbytes )

		cuda_driver.memset_d32( long(self.dpl) , 0 , 256 )

		mod = cuda_driver.module_from_file( 'jelly_kernel.cubin' )
		self.control = mod.get_function("control")
		self.control.prepare( "PPffff" )

		self.update_forces = self.jelly.update_forces

	def forces( self , dt , p = None ) :
		if p == None : p = self.pos
		return self.forces_cu( dt , p )

	def forces_cu( self , dt , p ) :
		cuda_driver.memset_d32( long(self.dnl) , 0 , 256 )
		cuda_driver.memset_d32( long(self.dfrs) , 0 , 256 )

		mpts = cuda_gl.BufferObject( long( self.gpts ) )
		dpts = mpts.map()

		self.control.prepared_call( self.grid , self.block , 
				dpts.device_ptr() , self.dnl , p[0] , p[1] , p[2] , self.l0 )

		self.update_forces.prepared_call( self.grid , self.block , 
				self.dfrs , self.dnl , self.dpl , self.K , self.C , dt )

		dpts.unmap()
		mpts.unregister()

		return self.dfrs

	def forces_c( self , dt , p  ) :
		self.nl.fill(0)
		self.frs.fill(0)
		cjelly.control( self.pts , self.nl , p , self.l0 )
		cjelly.update_forces( self.frs , self.nl , self.pl , self.K , self.C , dt )
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

