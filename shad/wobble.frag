#version 330

uniform vec4 color;
//uniform vec3 lpos;

in vec3 normal;

out vec4 out_color;

void main() {
	vec3 ldir = vec3(1,1,0);
	float ndl = max(dot(normal, ldir), 0.0);
	out_color = (ndl*.5 + .5) * color; 
}

// vim: ft=glsl:

