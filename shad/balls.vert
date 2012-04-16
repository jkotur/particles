#version 330

layout(location = 0) in vec4 pts;

uniform mat4 modelview;
uniform mat4 projection;

void main()
{
	gl_Position = projection * modelview * pts;
}

// vim: ft=glsl:

