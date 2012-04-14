
#include "cumath.h"

#define uint int
#define real float
#define real3 float3
#define real4 float4
#define mkreal3 make_float3
#define mkreal4 make_float4

#define Gx  9
#define Gy -9
#define Gz  9

__device__ real norm( real4 p )
{
	return sqrt( p.x*p.x + p.y*p.y + p.z*p.z );
}
 
extern "C" {

__global__ void update_pts( real4*pts , real3*vel , real3*acc , real dt , uint num )
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i >= num ) return;

	real3 p = mkreal3(pts[i].x,pts[i].y,pts[i].z);

	p = p + vel[i]*dt + acc[i]*dt*dt;

	pts[i].x = p.x;
	pts[i].y = p.y;
	pts[i].z = p.z;
}

__global__ void update_vel( real3*vel , real3*acc , real3*frs , real*mas , real dt , uint num )
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i >= num ) return;

	real3 v = mkreal3(0,0,0);
	real3 a = mkreal3(0,0,0);

	v = vel[i] + acc[i]*dt/2.0;
	a = frs[i] / mas[i];
	v = v      + a     *dt/2.0;

	acc[i] = a;
	vel[i] = v;
}

__global__ void update_frs( real4*pts , real3*frs , uint num )
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i >= num ) return;

	frs[i].x = Gx;
	frs[i].y = Gy;
	frs[i].z = Gz;
}

__global__ void collisions( real4*pts , real3*vel , real3*acc , real dt , real*brd , real u , uint num )
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i >= num ) return;

	for( unsigned int j=0 ; j<3 ; ++j )
	{
		if( pts[i].x <= brd[0] || pts[i].x >= brd[1] )
		{
			real prv = pts[i].x - vel[i].x*dt - acc[i].x*dt*dt;
			pts[i].x = prv;
			vel[i].x *= -u;
		}
		if( pts[i].y <= brd[2] || pts[i].y >= brd[3] )
		{
			real prv = pts[i].y - vel[i].y*dt - acc[i].y*dt*dt;
			pts[i].y = prv;
			vel[i].y *= -u;
		}
		if( pts[i].z <= brd[4] || pts[i].z >= brd[5] )
		{
			real prv = pts[i].z - vel[i].z*dt - acc[i].z*dt*dt;
			pts[i].z = prv;
			vel[i].z *= -u;
		}
	}
}

}

