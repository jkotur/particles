
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

#define Q 19
#define TAU 1000.0

__constant__ real PI = 3.1415;
__constant__ real C  = 340.0;
__constant__ real W  = 1/TAU;

__device__ real feq( real3 c , real3 v )
{
	real C2 = C*C;
	real cv = dot(c,v);
	return 1 + 3*cv/C2 + 4.5*cv/(C2*C2) - 1.5*dot(v,v)/C2 ;
}

extern "C" {

__global__ void collision_step( real*pts , int dx , int dy , int dz )
{
	uint x = threadIdx.x + blockDim.x * blockIdx.x;
	uint y = threadIdx.y + blockDim.y * blockIdx.y;
	uint z = threadIdx.z + blockDim.z * blockIdx.z;
	if( x >= dx || y >= dy || z >= dz ) return;
	uint i = ((( x*dy ) + y )*dz + z)*Q;

	real g = 0.0f;
	for( uint j=0 ; j<Q ; ++j ) g += pts[i+j];

	real3 V = mkreal3(0,0,0);

	V += pts[i+ 1] * mkreal3( 1, 0, 0);
	V += pts[i+ 2] * mkreal3( 0, 1, 0);
	V += pts[i+ 3] * mkreal3( 0, 0, 1);
	V += pts[i+ 4] * mkreal3(-1, 0, 0);
	V += pts[i+ 5] * mkreal3( 0,-1, 0);
	V += pts[i+ 6] * mkreal3( 0, 0,-1);
	V += pts[i+ 7] * mkreal3( 0, 1, 1);
	V += pts[i+ 8] * mkreal3( 1, 0, 1);
	V += pts[i+ 9] * mkreal3( 1, 1, 0);
	V += pts[i+10] * mkreal3( 0,-1,-1);
	V += pts[i+11] * mkreal3(-1, 0,-1);
	V += pts[i+12] * mkreal3(-1,-1, 0);
	V += pts[i+13] * mkreal3( 0,+1,-1);
	V += pts[i+14] * mkreal3(+1, 0,-1);
	V += pts[i+15] * mkreal3(+1,-1, 0);
	V += pts[i+16] * mkreal3( 0,-1,+1);
	V += pts[i+17] * mkreal3(-1, 0,+1);
	V += pts[i+18] * mkreal3(-1,+1, 0);

	if( g < 1e-5 && g > -1e-5 ) return;

	V = V / g;

	pts[i   ] = pts[i   ]*(1.0-W) + W*g*feq( mkreal3( 0, 0, 0), V )/ 3.0;
	pts[i+ 1] = pts[i+ 1]*(1.0-W) + W*g*feq( mkreal3( 1, 0, 0), V )/18.0;
    pts[i+ 2] = pts[i+ 2]*(1.0-W) + W*g*feq( mkreal3( 0, 1, 0), V )/18.0;
    pts[i+ 3] = pts[i+ 3]*(1.0-W) + W*g*feq( mkreal3( 0, 0, 1), V )/18.0;
    pts[i+ 4] = pts[i+ 4]*(1.0-W) + W*g*feq( mkreal3(-1, 0, 0), V )/18.0;
    pts[i+ 5] = pts[i+ 5]*(1.0-W) + W*g*feq( mkreal3( 0,-1, 0), V )/18.0;
    pts[i+ 6] = pts[i+ 6]*(1.0-W) + W*g*feq( mkreal3( 0, 0,-1), V )/18.0;
    pts[i+ 7] = pts[i+ 7]*(1.0-W) + W*g*feq( mkreal3( 0, 1, 1), V )/18.0;
    pts[i+ 8] = pts[i+ 8]*(1.0-W) + W*g*feq( mkreal3( 1, 0, 1), V )/36.0;
    pts[i+ 9] = pts[i+ 9]*(1.0-W) + W*g*feq( mkreal3( 1, 1, 0), V )/36.0;
    pts[i+10] = pts[i+10]*(1.0-W) + W*g*feq( mkreal3( 0,-1,-1), V )/36.0;
    pts[i+11] = pts[i+11]*(1.0-W) + W*g*feq( mkreal3(-1, 0,-1), V )/36.0;
    pts[i+12] = pts[i+12]*(1.0-W) + W*g*feq( mkreal3(-1,-1, 0), V )/36.0;
    pts[i+13] = pts[i+13]*(1.0-W) + W*g*feq( mkreal3( 0,+1,-1), V )/36.0;
    pts[i+14] = pts[i+14]*(1.0-W) + W*g*feq( mkreal3(+1, 0,-1), V )/36.0;
    pts[i+15] = pts[i+15]*(1.0-W) + W*g*feq( mkreal3(+1,-1, 0), V )/36.0;
    pts[i+16] = pts[i+16]*(1.0-W) + W*g*feq( mkreal3( 0,-1,+1), V )/36.0;
    pts[i+17] = pts[i+17]*(1.0-W) + W*g*feq( mkreal3(-1, 0,+1), V )/36.0;
    pts[i+18] = pts[i+18]*(1.0-W) + W*g*feq( mkreal3(-1,+1, 0), V )/36.0;

	if( x == 0 ) {
		pts[i+ 1] += pts[i+ 4];
		pts[i+14] += pts[i+11];
		pts[i+15] += pts[i+12];
		pts[i+ 8] += pts[i+17];
		pts[i+ 9] += pts[i+18];
		pts[i+ 4]  = 0;
		pts[i+11]  = 0;
		pts[i+12]  = 0;
		pts[i+17]  = 0;
		pts[i+18]  = 0;
	}
	if( x == dx-1 ) {
		pts[i+ 4] += pts[i+ 1];
		pts[i+11] += pts[i+14];
		pts[i+12] += pts[i+15];
		pts[i+17] += pts[i+ 8];
		pts[i+18] += pts[i+ 9];
		pts[i+ 1]  = 0;
		pts[i+14]  = 0;
		pts[i+15]  = 0;
		pts[i+ 8]  = 0;
		pts[i+ 9]  = 0;
	}

	if( y == 0 ) {
		pts[i+ 2] += pts[i+ 5];
		pts[i+13] += pts[i+10];
		pts[i+18] += pts[i+12];
		pts[i+ 9] += pts[i+15];
		pts[i+ 7] += pts[i+16];
		pts[i+ 5]  = 0;
		pts[i+10]  = 0;
		pts[i+12]  = 0;
		pts[i+15]  = 0;
		pts[i+16]  = 0;
	}
	if( y == dy-1 ) {
		pts[i+ 5] += pts[i+ 2];
		pts[i+10] += pts[i+13];
		pts[i+12] += pts[i+18];
		pts[i+15] += pts[i+ 9];
		pts[i+16] += pts[i+ 7];
		pts[i+ 2]  = 0;
		pts[i+13]  = 0;
		pts[i+18]  = 0;
		pts[i+ 9]  = 0;
		pts[i+ 7]  = 0;
	}

	if( z == 0 ) {
		pts[i+ 3] += pts[i+ 6];
		pts[i+16] += pts[i+10];
		pts[i+17] += pts[i+11];
		pts[i+ 7] += pts[i+13];
		pts[i+ 8] += pts[i+14];
		pts[i+ 6]  = 0;
		pts[i+10]  = 0;
		pts[i+11]  = 0;
		pts[i+13]  = 0;
		pts[i+14]  = 0;
	}
	if( z == dz-1 ) {
		pts[i+ 6] += pts[i+ 3];
		pts[i+10] += pts[i+16];
		pts[i+11] += pts[i+17];
		pts[i+13] += pts[i+ 7];
		pts[i+14] += pts[i+ 8];
		pts[i+ 3]  = 0;
		pts[i+16]  = 0;
		pts[i+17]  = 0;
		pts[i+ 7]  = 0;
		pts[i+ 8]  = 0;
	}
}

__global__ void streaming_step( real*f1 , real*f2 , int Dx , int Dy , int Dz )
{
	uint x = threadIdx.x + blockDim.x * blockIdx.x;
	uint y = threadIdx.y + blockDim.y * blockIdx.y;
	uint z = threadIdx.z + blockDim.z * blockIdx.z;
	if( x >= Dx || y >= Dy || z >= Dz ) return;
	uint i = ((( x*Dy ) + y )*Dz + z)*Q;

	uint dx = Dy * Dz * Q;
	uint dy =      Dz * Q;
	uint dz =           Q;

	f2[i            ] = f1[i   ];
	f2[i+dx      + 1] = f1[i+ 1];
	f2[i   +dy   + 2] = f1[i+ 2];
	f2[i      +dz+ 3] = f1[i+ 3];
	f2[i-dx      + 4] = f1[i+ 4];
	f2[i   -dy   + 5] = f1[i+ 5];
	f2[i      -dz+ 6] = f1[i+ 6];
	f2[i   +dy+dz+ 7] = f1[i+ 7];
	f2[i+dx   +dz+ 8] = f1[i+ 8];
	f2[i+dx+dy   + 9] = f1[i+ 9];
	f2[i   -dy-dz+10] = f1[i+10];
	f2[i-dx   -dz+11] = f1[i+11];
	f2[i-dx-dy   +12] = f1[i+12];
	f2[i   +dy-dz+13] = f1[i+13];
	f2[i+dx   -dz+14] = f1[i+14];
	f2[i+dx-dy   +15] = f1[i+15];
	f2[i   -dy+dz+16] = f1[i+16];
	f2[i-dx   +dz+17] = f1[i+17];
	f2[i-dx+dy   +18] = f1[i+18];
/** for D3Q27 model **/
/*    f2[i+dx+dy+dz]*/
/*    f2[i-dx+dy+dz]*/
/*    f2[i+dx-dy+dz]*/
/*    f2[i+dx+dy-dz]*/
/*    f2[i+dx-dy-dz]*/
/*    f2[i-dx+dy-dz]*/
/*    f2[i-dx-dy+dz]*/
/*    f2[i-dx-dy-dz]*/
}

__global__ void colors( real4*col , real*pts , int dx , int dy , int dz )
{
	uint x = threadIdx.x + blockDim.x * blockIdx.x;
	uint y = threadIdx.y + blockDim.y * blockIdx.y;
	uint z = threadIdx.z + blockDim.z * blockIdx.z;
	if( x >= dx || y >= dy || z >= dz ) return;
	uint i = (( x*dy ) + y )*dz + z;
	uint iQ = i*Q;

	real g = 0.0f;
	for( uint j=0 ; j<Q ; ++j ) g += pts[iQ+j];

	col[i].w = g > .001 ? 1.0 : 0.0;
}

}

