
#include "cumath.h"

#define uint int
#define real float
#define real3 float3
#define real4 float4
#define mkreal3 make_float3
#define mkreal4 make_float4

#define Gx  0
#define Gy -9
#define Gz  0

__constant__ real PI = 3.1415;
__constant__ real h  = 2;
__constant__ real h2 = 4;
__constant__ real h6 = 64;
__constant__ real h9 = 512;

__constant__ real k  = 0.5;
__constant__ real d0 = 100; // rest density
__constant__ real p0 = 200; // rest pressure
__constant__ real vi = 10e0; // fluid viscosity

// interpolation kernel
__device__ real W( real3 r )
{
	real d = length(r);
	if( d > h ) return 0.0;
	return 315.0f/(64.0f*PI*h9)*pow((h2-d*d),3.0f);
}
 
__device__ real3 dW( real3 r )
{
	real d = length(r);
	if( d > h ) return mkreal3(0,0,0);
	return -45.0f/(PI*h6)*pow((h-d),2.0f)*r/d;
}
 
__device__ real ddW( real3 r )
{
	real d = length(r);
	if( d > h ) return 0.0;
	return 45.0f/(PI*h6)*(h-d);
}
 
// calculates pressure from density
__device__ real dtop( real d )
{
	return p0 + k*(d - d0);
}

extern "C" {

__global__ void update_pts( real3*pts , real3*vel , real3*acc , real dt , uint num )
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i >= num ) return;

	real3 p = mkreal3(pts[i].x,pts[i].y,pts[i].z);

	p = p + vel[i]*dt + acc[i]*dt*dt;

	pts[i].x = p.x;
	pts[i].y = p.y;
	pts[i].z = p.z;
}

__global__ void update_vel( real3*vel , real3*acc , real3*frs , real dt , uint num )
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i >= num ) return;

	real3 v = mkreal3(0,0,0);
	real3 a = mkreal3(0,0,0);

	v = vel[i] + acc[i]*dt/2.0;
	a = frs[i];
	v = v      + a     *dt/2.0;

	acc[i] = a;
	vel[i] = v;
}

__global__ void update_dns( real3*pts , real3*dns , real*mas , uint num )
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i >= num ) return;

	real3 p = pts[i];

	real d = 0;
	for( int j=0; j<num; ++j )
	{
		if( j==i ) continue;
		d += mas[j] * W( p - pts[j] );
	}

}

__global__ void update_frs( real3*pts , real3*vel , real3*frs , real*mas , real*dns , uint num )
{
	uint i = threadIdx.x + blockDim.x * blockIdx.x;
	if( i >= num ) return;

	real3 ptsi = pts[i];
	real  prsi = dtop( dns[i] );
	real3 veli = vel[i];

	real3 a = mkreal3(0,0,0);
	for( int j=0; j<num; ++j )
	{
		if( j==i ) continue;
		real prsj = dtop( dns[j] );
		a += - mas[j]*(prsi+prsj)/(2*dns[j])*dW( ptsi - pts[j] );
		a += vi * mas[j]*(vel[j]-veli)/dns[j]*ddW( ptsi - pts[j] );
	}

	frs[i].x = a.x + Gx;
	frs[i].y = a.y + Gy;
	frs[i].z = a.z + Gz;
}

__global__ void collisions( real3*pts , real3*vel , real3*acc , real dt , real*brd , real u , uint num )
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

