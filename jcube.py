
import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB import *
from OpenGL.GL.EXT import *

import math as m
import numpy as np
import random as rnd

import shaders as sh

import cjelly
from drawable import Drawable

class JellyCube( Drawable ) :
	SHAPE = (4,4,4)
	C = 50.0
	K = 2.0
	U = .5

	def __init__( self ) :
		self.pts = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.prv = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.pl  = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.nl  = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.frs = np.zeros( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2],4) , np.float32 )
		self.mas = np.ones ( (self.SHAPE[0],self.SHAPE[1],self.SHAPE[2]) , np.float32 )
		self.l0 = np.float32( 1 )
		self.l1 = m.sqrt(2) * self.l0

		self.__func( self.pts , lambda x,y,z : np.array((x,y,z,0)) * self.l0 )
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

	def set_borders( self , b ) :
		self.borders = np.array( b , np.float32 )

	def draw( self , mesh = None ) :
		if mesh == None :
			self.draw_pts() 
		else :
#            self.draw_pts() 
			self.draw_mesh( mesh )

	def draw_pts( self ) :
		glPointSize( 5 )
		glColor3f(1,0,0)
		glBegin(GL_POINTS)
		for x in range(self.SHAPE[0]) :
			for y in range(self.SHAPE[1]) :
				for z in range(self.SHAPE[2]) :
					glVertex3f( self.pts[x,y,z,0] , self.pts[x,y,z,1] , self.pts[x,y,z,2] )
		glEnd()

	def gfx_init( self ) :
		try :
			self.prog = sh.compile_program_vf( 'shad/wobble' )

			self.loc_mmv = sh.get_loc(self.prog,'modelview' )
			self.loc_mp  = sh.get_loc(self.prog,'projection')
			self.l_color = sh.get_loc(self.prog,'color'     )
			pointsid     = sh.get_loc(self.prog,'points'    )

		except ValueError as ve :
			print "Shader compilation failed: " + str(ve)
			sys.exit(0)    

		self.bid  = glGenBuffers(1) 
		self.btex = glGenTextures(1)

		glUseProgram( self.prog )
		glUniform1i( pointsid , 0 );
		glUseProgram( 0 )
		
	def draw_mesh( self , mesh ) :
		glBindBuffer(GL_ARRAY_BUFFER,self.bid)
		glBufferData(GL_ARRAY_BUFFER,self.pts,GL_STATIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER,0)

		glBindTexture(GL_TEXTURE_BUFFER,self.btex)
		glTexBuffer(GL_TEXTURE_BUFFER,GL_RGBA32F,self.bid)
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


	def wobble( self , dt , addfrs = None ) :
		self.nl.fill(0)
		self.frs.fill(0)

		cjelly.springs( self.pts , self.nl , self.l0 , self.l1 )

		cjelly.update_forces( self.frs , self.nl , self.pl , self.K , self.C , dt )
		if addfrs != None : self.frs += addfrs

		cjelly.update( self.pts , self.prv , self.frs , self.mas , self.K ,self.C , dt )

		cjelly.collisions( self.pts , self.prv , self.borders , self.U )

