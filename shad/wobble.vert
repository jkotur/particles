#version 330

layout(location = 0) in vec4 uvw;

uniform mat4 modelview;
uniform mat4 projection;

uniform samplerBuffer points;

float tmp[4];
float b[12];

float decasteljau( float t , float pts[4] )
{
	int len = 4;

	for( int i=0 ; i<len ; ++i )
		tmp[i] = 0.0;

	for( int i=len-1 ; i>=0 ; --i )
		tmp[i] = pts[i];

	float t1 = 1.0-t;
	for( int k=len-1 ; k>0 ; --k )
		for( int i=0 ; i<k ; ++i )
			tmp[i] = tmp[i]*t1 + tmp[i+1]*t;
	return tmp[0];
}

void main()
{
	float pts[4];
	pts[0] = 1;
	pts[1] = 0;
	pts[2] = 0;
	pts[3] = 0;
	b[ 0] = decasteljau( uvw.x , pts );
	b[ 1] = decasteljau( uvw.y , pts );
	b[ 2] = decasteljau( uvw.z , pts );
	pts[0] = 0;
	pts[1] = 1;
	b[ 3] = decasteljau( uvw.x , pts );
	b[ 4] = decasteljau( uvw.y , pts );
	b[ 5] = decasteljau( uvw.z , pts );
	pts[1] = 0;
	pts[2] = 1;
	b[ 6] = decasteljau( uvw.x , pts );
	b[ 7] = decasteljau( uvw.y , pts );
	b[ 8] = decasteljau( uvw.z , pts );
	pts[2] = 0;
	pts[3] = 1;
	b[ 9] = decasteljau( uvw.x , pts );
	b[10] = decasteljau( uvw.y , pts );
	b[11] = decasteljau( uvw.z , pts );

	vec4 pos = vec4( 0 , 0 , 0 , 1 );
	vec4 ptn = vec4( 0 , 0 , 0 , 0 );

	for( int x=0 ; x<4 ; ++x ) 
		for( int y=0 ; y<4 ; ++y )
			for( int z=0 ; z<4 ; ++z )
			{
				ptn = texelFetch( points , 16*x+y*4+z );
				pos += ptn*b[3*x  ]*b[3*y+1]*b[3*z+2];
			}

	gl_Position = projection * modelview * pos;
}

// vim: ft=glsl:

