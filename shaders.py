import sys

from OpenGL.GL import *
from OpenGL.GL.ARB.geometry_shader4 import *
from OpenGL.GL.EXT.geometry_shader4 import *
#from OpenGL.GL.ARB.shader_objects import *

def get_loc( prog , loc ) :                                   
	location = glGetUniformLocation(prog,loc)
	if location in (None,-1): raise ValueError ('No uniform: ' + loc)
	return location        

def compile_shader(source, shader_type):
	shader = glCreateShader(shader_type)
	glShaderSource(shader, source )
	glCompileShader(shader)

	status = 0
	status = glGetShaderiv(shader, GL_COMPILE_STATUS)
	if status == GL_FALSE:
		print_shader_log(shader,shader_type)
		glDeleteShader(shader)
		raise ValueError, 'Shader compilation failed'
	return shader

def compile_program_vf(name) :
	return compile_program_from_file(name+".vert",name+".frag")

def compile_program_vfg(name) :
	return compile_program_from_file(name+".vert",name+".frag",name+".geom")

def compile_program_from_file(vertex_file , fragment_file , geometry_file = None ) :
	vsrc = open(vertex_file,"r")
	fsrc = open(fragment_file,"r")
	gsrc = open(geometry_file,"r") if geometry_file else None

	return compile_program_from_source(vsrc.read(),fsrc.read(),gsrc.read() if gsrc else None)

def compile_program_from_source(vertex_source, fragment_source , geometry_source = None ) :
	vertex_shader = None
	geometry_shader = None
	fragment_shader = None
	program = glCreateProgram()

	if vertex_source:
		vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
		glAttachShader(program, vertex_shader)
	if fragment_source:
		fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
		glAttachShader(program, fragment_shader)
	if geometry_source:
		geometry_shader = compile_shader(geometry_source, GL_GEOMETRY_SHADER)
		glAttachShader(program, geometry_shader)

	glLinkProgram(program)

	if vertex_shader:
		glDeleteShader(vertex_shader)
	if fragment_shader:
		glDeleteShader(fragment_shader)
	if geometry_shader:
		glDeleteShader(geometry_shader)

	status = 0
	status = glGetProgramiv(program, GL_LINK_STATUS)
	if status == GL_FALSE:
		print_program_log( program )
		glDeleteProgram(program)
		raise ValueError, 'Program compilation failed'

	return program

def print_program_log(program):
	log = glGetProgramInfoLog(program)
	print >> sys.stderr 
	print >> sys.stderr , ">** program error **"
	print >> sys.stderr , "'",log,"'"
	print >> sys.stderr , "<** program error **"
	print >> sys.stderr

def print_shader_log(shader,info):
	log = glGetShaderInfoLog(shader)
	print >> sys.stderr 
	print >> sys.stderr , ">** shader " + str(info) + " error **"
	print >> sys.stderr , "'",log,"'"
	print >> sys.stderr , "<** shader error **"
	print >> sys.stderr
#    print ">** shader error **"
#    print "'",log,"'"
#    print "<** shader error **"
#    print

