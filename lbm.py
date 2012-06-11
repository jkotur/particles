
import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB import *
from OpenGL.GL.EXT import *

import pycuda.autoinit
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda.gpuarray as cuda_garr

import math as m
import numpy as np
import random as rnd

import shaders as sh

from drawable import Drawable

class LBM( Drawable ) :
	BOX = int(23)
	NUM = BOX**3
	D = int(3)
	Q = int(19)
	TAU = 0.5

	BALLSIZE = 15

	def __init__( self ) :
		self.f = np.zeros( (self.BOX,self.BOX,self.BOX,self.Q), np.float32 )
		self.pos = np.zeros( (self.BOX,self.BOX,self.BOX,4), np.float32 )

		self._init_ball( self.BALLSIZE )
		self._init_pos()

		self.gpos = None

	def _init_ball( self , p ) :
		b2 = self.BOX / 2.0
		for x in range(self.BOX) :
			for y in range(self.BOX) :
				for z in range(self.BOX) :
					for w in range(self.Q) :
						self.f[x,y,z,w] = 0
					if (x-b2)**2 + (y-b2)**2 + (z-b2)**2 < p :
						self.f[x,y,z,0] = .1

	def _init_pos( self ) :
		b2 = self.BOX / 2.0
		for x in range(self.BOX) :
			for y in range(self.BOX) :
				for z in range(self.BOX) :
					self.pos[x,y,z] = np.array((x-b2,y-b2,z-b2,self.f[x,y,z,0]),np.float32)

	def set_borders( self , b ) :
		self.borders = np.array( b , np.float32 )

	def draw( self ) :
		self.draw_balls() 

	def draw_pts( self ) :
		glPointSize( 5 )
		glColor3f(1,0,0)
		glEnableClientState(GL_VERTEX_ARRAY)
		glBindBuffer( GL_ARRAY_BUFFER , self.gpos )
		glVertexPointer( 4 , GL_FLOAT , 0 , None )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )
		glDrawArrays( GL_POINTS , 0 , self.NUM )
		glDisableClientState(GL_VERTEX_ARRAY)

	def draw_balls( self ) :
		glPointSize( 5 )
		glPushMatrix()

		glAlphaFunc( GL_GREATER , 0.1 );
		glEnable( GL_ALPHA_TEST )

		glUseProgram( self.prog )

		mmv = glGetFloatv(GL_MODELVIEW_MATRIX)   
		mp  = glGetFloatv(GL_PROJECTION_MATRIX)  
												 
		glUniformMatrix4fv(self.loc_mmv,1,GL_FALSE,mmv)
		glUniformMatrix4fv(self.loc_mp ,1,GL_FALSE,mp )

		glUniform4f(self.l_color , 1 , .5 , 0 , 1 )
		glUniform1f(self.l_size  , 1.5 )

		glEnableClientState(GL_VERTEX_ARRAY)
		glBindBuffer( GL_ARRAY_BUFFER , self.gpos )
		glVertexPointer( 4 , GL_FLOAT , 0 , None )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )
		glDrawArrays( GL_POINTS , 0 , self.NUM )
		glDisableClientState(GL_VERTEX_ARRAY)

		glDisable( GL_ALPHA_TEST )
		glUseProgram( 0 )
		glPopMatrix()

	def gfx_init( self ) :
		try :
			print 'compiling'
			self.prog = sh.compile_program_vfg( 'shad/balls' )

			print 'compiled'

			self.loc_mmv = sh.get_loc(self.prog,'modelview' )
			self.loc_mp  = sh.get_loc(self.prog,'projection')
			self.l_color = sh.get_loc(self.prog,'color'     )
			self.l_size  = sh.get_loc(self.prog,'ballsize'  )

		except ValueError as ve :
			print "Shader compilation failed: " + str(ve)
			sys.exit(0)    

#        glUseProgram( self.prog )
#        glUniform1i( pointsid , 0 );
#        glUseProgram( 0 )

		#
		# cuda init
		#
		self.grid = (int(self.BOX),int(self.BOX))
		self.block = (1,1,int(self.BOX))

		print 'CUDA: block %s , grid %s' % (str(self.block),str(self.grid))
#        print cuda_driver.device_attribute.MAX_THREADS_PER_BLOCK
#        print cuda_driver.device_attribute.MAX_BLOCK_DIM_X
#        print cuda_driver.device_attribute.MAX_BLOCK_DIM_Y
#        print cuda_driver.device_attribute.MAX_BLOCK_DIM_Z

		floatbytes = np.dtype(np.float32).itemsize

		self.gpos = glGenBuffers(1)
		glBindBuffer( GL_ARRAY_BUFFER , self.gpos )
		glBufferData( GL_ARRAY_BUFFER , self.pos.nbytes, self.pos, GL_STREAM_DRAW )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

		self.df1 = cuda_driver.mem_alloc( self.f.nbytes )
		self.df2 = cuda_driver.mem_alloc( self.f.nbytes )

		cuda_driver.memcpy_htod( self.df1 , self.f )
		cuda_driver.memset_d32( self.df2 , 0 , self.NUM*self.Q )

		mod = cuda_driver.module_from_file( 'lbm_kernel.cubin' )

		self.collision = mod.get_function("collision_step")
		self.collision.prepare( "Piii" )

		self.streaming = mod.get_function("streaming_step")
		self.streaming.prepare( "PPiii" )

		self.colors = mod.get_function("colors")
		self.colors.prepare( "PPiii" )

	def wave( self , dt ) :
		self.wave_cu( dt )

	def wave_cu( self , dt ) :
		mpos = cuda_gl.BufferObject( long( self.gpos ) )
		dpos = mpos.map()

#        self._debug_print()

		self.collision.prepared_call( self.grid , self.block ,
			self.df1 , self.BOX , self.BOX , self.BOX )

#        self._debug_print()

		self.streaming.prepared_call( self.grid , self.block ,
			self.df1 , self.df2 , self.BOX , self.BOX , self.BOX )

		cuda_driver.memcpy_dtod( self.df1 , self.df2 , self.f.nbytes )

		self.colors.prepared_call( self.grid , self.block ,
			dpos.device_ptr() , self.df1 , self.BOX , self.BOX , self.BOX )

		dpos.unmap()
		mpos.unregister()

	def _debug_print( self ) :
		cuda_driver.memcpy_dtoh( self.f , self.df1 )

		np.set_printoptions( 3 , 10000 , linewidth = 200 , suppress = True )

		print '#'*80
		print self.f

