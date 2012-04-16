#version 330

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform float ballsize;

out vec2 coord;

void main() {
//    vec3 v1 = gl_in[0].gl_Position.xyz;
//    normal = normalize( cross(v2,v1) );

	gl_Position = gl_in[0].gl_Position + vec4(-.5,-.5,0,0)*ballsize;
	coord = vec2(-1,-1);
	EmitVertex();
	gl_Position = gl_in[0].gl_Position + vec4( .5,-.5,0,0)*ballsize;
	coord = vec2( 1,-1);
	EmitVertex();
	gl_Position = gl_in[0].gl_Position + vec4(-.5, .5,0,0)*ballsize;
	coord = vec2(-1, 1);
	EmitVertex();
	gl_Position = gl_in[0].gl_Position + vec4( .5, .5,0,0)*ballsize;
	coord = vec2( 1, 1);
	EmitVertex();
	EndPrimitive();
}


// vim: ft=glsl:

