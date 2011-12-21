
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

import cjelly
from drawable import Drawable

class JellyCube( Drawable ) :
	SHAPE = (4,4,4)
	C = 2000.0
	K = 50.0
	U = .5

	def __init__( self ) :
		self.pts = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.prv = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.pl  = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.nl  = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.frs = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.mas = np.ones ( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2]) , np.float32 )
		self.l0 = np.float32( 1 )
		self.l1 = np.float32( m.sqrt(2) * self.l0 )

		self.__func( self.pts , lambda x,y,z : np.array((x,y,z,1)) * self.l0 )
		self.__func( self.prv , lambda x,y,z : self.pts[x,y,z] )
		self.__func( self.pl  , lambda x,y,z : 0 )
		self.__func( self.mas , lambda x,y,z : 10 )
		self.pts[2,3,2,1] += .2
		self.prv[2,3,2,1] += .2
		self.pts[1,3,1,1] -= .2
		self.prv[1,3,1,1] -= .2

		self.dbrd = None
		self.ctlpos = None

	def __func( self , p , f ) :
		for x in range(self.SHAPE[0]) :
			for y in range(self.SHAPE[1]) :
				for z in range(self.SHAPE[2]) :
					p[x,y,z] = f( x,y,z )

	def set_borders( self , b ) :
		self.borders = np.array( b , np.float32 )
		if self.dbrd == None :
			self.dbrd = cuda_driver.mem_alloc( self.borders.nbytes )
		cuda_driver.memcpy_htod( self.dbrd , self.borders )

	def draw( self , mesh = None ) :
		if mesh == None :
			self.draw_pts() 
		else :
			self.draw_mesh( mesh )

	def draw_pts( self ) :
		glPointSize( 5 )
		glColor3f(1,0,0)
		glEnableClientState(GL_VERTEX_ARRAY)
		glBindBuffer( GL_ARRAY_BUFFER , self.gpts )
		glVertexPointer( 4 , GL_FLOAT , 0 , None )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )
		glDrawArrays( GL_POINTS , 0 , 64 )
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
		self.grid = (1,1)
		self.block = (4,4,4)

		self.gpts = glGenBuffers(1)
		glBindBuffer( GL_ARRAY_BUFFER , self.gpts )
		glBufferData( GL_ARRAY_BUFFER , self.pts.nbytes , self.pts , GL_STREAM_DRAW )
		glBindBuffer( GL_ARRAY_BUFFER , 0 )

		self.dprv = cuda_driver.mem_alloc( self.prv.nbytes )
		self.dfrs = cuda_driver.mem_alloc( self.frs.nbytes )
		self.dnl  = cuda_driver.mem_alloc( self.nl .nbytes )
		self.dpl  = cuda_driver.mem_alloc( self.pl .nbytes )
		self.dmas = cuda_driver.mem_alloc( self.mas.nbytes )

		cuda_driver.memcpy_htod( self.dprv , self.prv )
		cuda_driver.memcpy_htod( self.dfrs , self.frs )
		cuda_driver.memcpy_htod( self.dnl  , self.nl  )
		cuda_driver.memcpy_htod( self.dpl  , self.pl  )
		cuda_driver.memcpy_htod( self.dmas , self.mas )

		mod = cuda_driver.module_from_file( 'jelly_kernel.cubin' )

		self.springs = mod.get_function("springs")
		self.springs.prepare( "PPff" )
		
		self.update_forces = mod.get_function("update_forces")
		self.update_forces.prepare( "PPPfff" )
		
		self.update = mod.get_function("update")
		self.update.prepare( "PPPPfff" )

		self.update2 = mod.get_function("update2")
		self.update2.prepare( "PPPPPfff" )

		self.collisions = mod.get_function("collisions")
		self.collisions.prepare( "PPPf" )

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


	def wobble( self , dt , addfrs = None ) :
		self.wobble_cu( dt , addfrs )

	def wobble_cu( self , dt , addfrs = None ) :
		cuda_driver.memset_d32( long(self.dnl) , 0 , 256 )
		cuda_driver.memset_d32( long(self.dfrs) , 0 , 256 )

		mpts = cuda_gl.BufferObject( long( self.gpts ) )
		dpts = mpts.map()

#        cjelly.springs( self.pts , self.nl , self.l0 , self.l1 )

		self.springs.prepared_call( self.grid , self.block , 
				dpts.device_ptr() , self.dnl , self.l0 , self.l1 )

		self.update_forces.prepared_call( self.grid , self.block , 
				self.dfrs , self.dnl , self.dpl , self.K , self.C , dt )
		if addfrs != None :
			self.update2.prepared_call( self.grid , self.block ,
				dpts.device_ptr() , self.dprv , self.dfrs , addfrs , self.dmas , self.K , self.C , dt )
		else :
			self.update.prepared_call( self.grid , self.block ,
				dpts.device_ptr() , self.dprv , self.dfrs , self.dmas , self.K , self.C , dt )


		self.collisions.prepared_call( self.grid , self.block ,
				dpts.device_ptr() , self.dprv , self.dbrd , self.U )

		dpts.unmap()
		mpts.unregister()

	def wobble_c( self , dt , addfrs = None ) :
		self.nl.fill(0)
		self.frs.fill(0)

		cjelly.update_forces( self.frs , self.nl , self.pl , self.K , self.C , dt )

		if addfrs != None : self.frs += addfrs

		cjelly.update( self.pts , self.prv , self.frs , self.mas , self.K ,self.C , dt )
		cjelly.collisions( self.pts , self.prv , self.borders , self.U )

