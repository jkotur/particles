
#include "cumath.h"

#define real float
#define real3 float3
#define real4 float4
#define mkreal4 make_float4

#define SHAPEX 4
#define SHAPEY 4
#define SHAPEZ 4

#define L(x,y,z) ((x) + ( (y)  + (z) * SHAPEY ) * SHAPEX)

#define Gx  0
#define Gy -9
#define Gz  0

__device__ real norm( real4 p )
{
	return sqrt( p.x*p.x + p.y*p.y + p.z*p.z );
}
 
__device__ real4 spring_force( real4 p0 , real4 p1 , real l0 )
{
	real4 dp = p1 - p0;
	real  dl = norm( dp );
	if( dl == 0 )
		return mkreal4(0,0,0,0);
	real l = l0 -dl;
	return dp / dl * l;
}

extern "C" {

__global__ void control( real4*pts , real4*nl , real ctlx , real ctly , real ctlz , real l )
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int z = threadIdx.z;

	int i = x + ( y  + z * SHAPEY ) * SHAPEX;

	real4 p = mkreal4(0,0,0,0);

	p.x = ctlx + l * (z-1.5);
	p.y = ctly + l * (y-1.5);
	p.z = ctlz + l * (x-1.5);
	if( !(x%3) && !(y%3) && !(z%3) )
		nl[i] += spring_force( pts[i] , p , 0 );
}

__global__ void springs( real4*pts , real4*nl , float l0 , float l1 )
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int z = threadIdx.z;

	int i = x + ( y  + z * SHAPEY ) * SHAPEX;

	if( x+1 < SHAPEX )               nl[i]+=spring_force(pts[i],pts[L(x+1,y,z)],l0);
	if( y+1 < SHAPEY )               nl[i]+=spring_force(pts[i],pts[L(x,y+1,z)],l0);
	if( z+1 < SHAPEZ )               nl[i]+=spring_force(pts[i],pts[L(x,y,z+1)],l0);
	if( x-1 >= 0 )                   nl[i]+=spring_force(pts[i],pts[L(x-1,y,z)],l0);
	if( y-1 >= 0 )                   nl[i]+=spring_force(pts[i],pts[L(x,y-1,z)],l0);
	if( z-1 >= 0 )                   nl[i]+=spring_force(pts[i],pts[L(x,y,z-1)],l0);
	if( x+1 < SHAPEX && y+1<SHAPEY ) nl[i]+=spring_force(pts[i],pts[L(x+1,y+1,z)],l1);
	if( y+1 < SHAPEY && z+1<SHAPEZ ) nl[i]+=spring_force(pts[i],pts[L(x,y+1,z+1)],l1);
	if( z+1 < SHAPEZ && x+1<SHAPEX ) nl[i]+=spring_force(pts[i],pts[L(x+1,y,z+1)],l1);
	if( x-1 >= 0 && y+1 < SHAPEY )   nl[i]+=spring_force(pts[i],pts[L(x-1,y+1,z)],l1);
	if( y-1 >= 0 && z+1 < SHAPEZ )   nl[i]+=spring_force(pts[i],pts[L(x,y-1,z+1)],l1);
	if( z-1 >= 0 && x+1 < SHAPEX )   nl[i]+=spring_force(pts[i],pts[L(x+1,y,z-1)],l1);
	if( x+1 < SHAPEX && y-1 >= 0 )   nl[i]+=spring_force(pts[i],pts[L(x+1,y-1,z)],l1);
	if( y+1 < SHAPEY && z-1 >= 0 )   nl[i]+=spring_force(pts[i],pts[L(x,y+1,z-1)],l1);
	if( z+1 < SHAPEZ && x-1 >= 0 )   nl[i]+=spring_force(pts[i],pts[L(x-1,y,z+1)],l1);
	if( x-1 >= 0 && y-1 >= 0 )       nl[i]+=spring_force(pts[i],pts[L(x-1,y-1,z)],l1);
	if( y-1 >= 0 && z-1 >= 0 )       nl[i]+=spring_force(pts[i],pts[L(x,y-1,z-1)],l1);
	if( z-1 >= 0 && x-1 >= 0 )       nl[i]+=spring_force(pts[i],pts[L(x-1,y,z-1)],l1);
}

__global__ void update_forces( real4*frs , real4*nl , real4*pl , real k , real c , real dt )
{
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;

	unsigned int i = x + ( y  + z * SHAPEY ) * SHAPEX;

	real4 lp = ( nl[i] - pl[i] ) / ( 2.0 * dt );
	frs[i] += -k * lp - c * nl[i];
	pl[i] = nl[i];
}


__global__ void update( real4*pts , real4*prv , real4*frs , real*mas , real k , real c , real dt )
{
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;

	unsigned int i = x + ( y  + z * SHAPEY ) * SHAPEX;

	real4 a = mkreal4(0,0,0,0);
	real4 n = mkreal4(0,0,0,1);

	real4 f = frs[i];
	f.x += Gx;
	f.y += Gy;
	f.z += Gz;
	a = f / mas[i];
	n = a * dt * dt + 2 * pts[i] - prv[i];
	prv[i] = pts[i];
	pts[i] = n;
}

__global__ void update2( real4*pts , real4*prv , real4*fr1 , real4*fr2 , real*mas , real k , real c , real dt )
{
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;

	unsigned int i = x + ( y  + z * SHAPEY ) * SHAPEX;

	real4 a = mkreal4(0,0,0,0);
	real4 n = mkreal4(0,0,0,1);

	real4 f = fr1[i] + fr2[i];
	f.x += Gx;
	f.y += Gy;
	f.z += Gz;
	a = f / mas[i];
	n = a * dt * dt + 2 * pts[i] - prv[i];
	prv[i] = pts[i];
	pts[i] = n;
	pts[i].w = 1.0;
}

__global__ void collisions( real4*pts , real4*prv , real*brd , real u )
{
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;

	unsigned int i = x + ( y  + z * SHAPEY ) * SHAPEX;

	real n;
	for( unsigned int j=0 ; j<3 ; ++j )
	{
		if( pts[i].x <= brd[0] || pts[i].x >= brd[1] )
		{
			n = prv[i].x+u*pts[i].x-u*prv[i].x;
			pts[i].x = prv[i].x;
			prv[i].x = n;
		}
		if( pts[i].y <= brd[2] || pts[i].y >= brd[3] )
		{
			n = prv[i].y+u*pts[i].y-u*prv[i].y;
			pts[i].y = prv[i].y;
			prv[i].y = n;
		}
		if( pts[i].z <= brd[4] || pts[i].z >= brd[5] )
		{
			n = prv[i].z+u*pts[i].z-u*prv[i].z;
			pts[i].z = prv[i].z;
			prv[i].z = n;
		}
	}
}

}

