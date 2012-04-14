
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

class Water( Drawable ) :
	NUM = int(8000)
	U = .5

	def __init__( self ) :
		self.pts = np.zeros( (self.NUM,4) , np.float32 )
#        self.prv = np.zeros( (self.NUM,4) , np.float32 )
		self.vel = np.zeros( (self.NUM,3) , np.float32 )
		self.acc = np.zeros( (self.NUM,3) , np.float32 )
		self.frs = np.zeros( (self.NUM,3) , np.float32 )
		self.mas = np.ones ( (self.NUM,1) , np.float32 )

		self.__func( self.pts , lambda i : np.array((i%20,int(i/20)%20,int(i/20/20)%20,10)) * 0.1 )
#        self.__func( self.prv , lambda i : self.pts[i] )
		self.__func( self.mas , lambda i : 10 )

		self.dbrd = None

	def __func( self , p , f ) :
		for i in range(self.NUM) :
			p[i] = f( i )

	def set_borders( self , b ) :
		self.borders = np.array( b , np.float32 )
		if self.dbrd == None :
			self.dbrd = cuda_driver.mem_alloc( self.borders.nbytes )
		cuda_driver.memcpy_htod( self.dbrd , self.borders )

	def draw( self ) :
		self.draw_pts() 

	def draw_pts( self ) :
		glPointSize( 5 )
		glColor3f(1,0,0)
		glEnableClientState(GL_VERTEX_ARRAY)
		glBindBuffer( GL_ARRAY_BUFFER , self.gpts )
		glVertexPointer( 4 , GL_FLOAT , 0 , None )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )
		glDrawArrays( GL_POINTS , 0 , self.NUM )
		glDisableClientState(GL_VERTEX_ARRAY)

	def gfx_init( self ) :
		try :
			print 'compiling'
			self.prog = sh.compile_program_vfg( 'shad/wobble' )

			print 'compiled'

			self.loc_mmv = sh.get_loc(self.prog,'modelview' )
			self.loc_mp  = sh.get_loc(self.prog,'projection')
			self.l_color = sh.get_loc(self.prog,'color'     )
			pointsid     = sh.get_loc(self.prog,'points'    )

		except ValueError as ve :
			print "Shader compilation failed: " + str(ve)
			sys.exit(0)    

		self.btex = glGenTextures(1)

		glUseProgram( self.prog )
		glUniform1i( pointsid , 0 );
		glUseProgram( 0 )

		#
		# cuda init
		#
		self.grid = (int(self.NUM/256)+1,1)
		self.block = (256,1,1)

		print 'CUDA: block %s , grid %s' % (str(self.block),str(self.grid))

		self.gpts = glGenBuffers(1)
		glBindBuffer( GL_ARRAY_BUFFER , self.gpts )
		glBufferData( GL_ARRAY_BUFFER , self.pts.nbytes , self.pts , GL_STREAM_DRAW )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

#        self.dprv = cuda_driver.mem_alloc( self.prv.nbytes )
		self.dvel = cuda_driver.mem_alloc( self.vel.nbytes )
		self.dacc = cuda_driver.mem_alloc( self.acc.nbytes )
		self.dfrs = cuda_driver.mem_alloc( self.frs.nbytes )
		self.dmas = cuda_driver.mem_alloc( self.mas.nbytes )

#        cuda_driver.memcpy_htod( self.dprv , self.prv )
		cuda_driver.memcpy_htod( self.dvel , self.vel )
		cuda_driver.memcpy_htod( self.dacc , self.acc )
		cuda_driver.memcpy_htod( self.dfrs , self.frs )
		cuda_driver.memcpy_htod( self.dmas , self.mas )

		mod = cuda_driver.module_from_file( 'water_kernel.cubin' )

		self.update_pts = mod.get_function("update_pts")
		self.update_pts.prepare( "PPPfi" )

		self.update_vel = mod.get_function("update_vel")
		self.update_vel.prepare( "PPPPfi" )

		self.update_frs = mod.get_function("update_frs")
		self.update_frs.prepare( "PPi" )

		self.collisions = mod.get_function("collisions")
		self.collisions.prepare( "PPPfPfi" )

	def draw_mesh( self , mesh ) :
		glPushMatrix()
		glScalef(2,2,2)
		glBindTexture(GL_TEXTURE_BUFFER,self.btex)
		glTexBuffer(GL_TEXTURE_BUFFER,GL_RGBA32F,self.gpts)
		glBindTexture(GL_TEXTURE_BUFFER,0)                    

		glUseProgram( self.prog )

		mmv = glGetFloatv(GL_MODELVIEW_MATRIX)   
		mp  = glGetFloatv(GL_PROJECTION_MATRIX)  
												 
		glUniformMatrix4fv(self.loc_mmv,1,GL_FALSE,mmv)
		glUniformMatrix4fv(self.loc_mp ,1,GL_FALSE,mp )

		glBindTexture(GL_TEXTURE_BUFFER,self.btex)

		glUniform4f(self.l_color , 1 , .5 , 0 , 1 )

		mesh.draw()

		glUseProgram( 0 )
		glPopMatrix()


	def wave( self , dt ) :
		self.wave_cu( dt )

	def wave_cu( self , dt ) :
		mpts = cuda_gl.BufferObject( long( self.gpts ) )
		dpts = mpts.map()

		self.update_pts.prepared_call( self.grid , self.block ,
			dpts.device_ptr() , self.dvel , self.dacc , dt , self.NUM )

		self.collisions.prepared_call( self.grid , self.block ,
				dpts.device_ptr() , self.dvel , self.dacc , dt , self.dbrd , self.U , self.NUM )

		self.update_frs.prepared_call( self.grid , self.block ,
			dpts.device_ptr() , self.dfrs , self.NUM )

		self.update_vel.prepared_call( self.grid , self.block ,
			self.dvel , self.dacc , self.dfrs , self.dmas , dt , self.NUM )

		dpts.unmap()
		mpts.unregister()

