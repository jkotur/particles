#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

out vec3 normal;

void main() {
	vec3 v1 = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	vec3 v2 = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
	normal = normalize( cross(v2,v1) );

	gl_Position = gl_in[0].gl_Position;
	EmitVertex();
	gl_Position = gl_in[1].gl_Position;
	EmitVertex();
	gl_Position = gl_in[2].gl_Position;
	EmitVertex();
	EndPrimitive();
}


// vim: ft=glsl:

