
import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB import *
from OpenGL.GL.EXT import *

import pycuda.autoinit
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda.gpuarray as cuda_garr

from sph import SPH
from lbm import LBM

from drawable import Drawable

class Water( Drawable ) :

	NONE_T , LBM_T , SPH_T = range(3)

	def __init__( self ) :
		self.sph = SPH()
		self.lbm = LBM()
		self.water = self.sph

	def set( self , water_type ) :
		if water_type == self.NONE_T :
			self.water = None
		elif water_type == self.SPH_T :
			self.water = self.sph
		elif water_type == self.LBM_T :
			self.water = self.lbm

	def set_borders( self , b ) :
		self.sph.set_borders( b )
		self.lbm.set_borders( b )

	def draw( self ) :
		self.water.draw()

	def gfx_init( self ) :
		self.sph.gfx_init()
		self.lbm.gfx_init()

	def wave( self , dt ) :
		self.water.wave( dt )

