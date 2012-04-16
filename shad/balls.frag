#version 330

uniform vec4 color;

in vec2 coord;

out vec4 out_color;

void main() {
	float xy = coord.x*coord.x + coord.y*coord.y;
	if( xy > 1 )
		out_color = vec4(0,0,0,0);
	else
	{
		vec3 normal = vec3(coord.x,coord.y,sqrt(1-xy));
		vec3 ldir = vec3(1,1,0);
		float ndl = max(dot(normal, ldir), 0.0);
		out_color = (ndl*.5 + .5) * color; 
	}
}

// vim: ft=glsl:

