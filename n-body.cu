#include <iostream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define N 40960
#define NAME "galaxy.tab"
#define TIME 0.02
#define G 0.0000000000667
#define M 10000000000000000
#define EPSI2 0.0001
#define XDIM 2000
#define YDIM 2000
#define ZDIM 2000
#define XMAX 38
#define YMAX 44 
#define ZMAX 38
#define velFactor 40.0f
#define MAXSTR 256

void draw(Mat bg, float4 pos, int r0, int rmin);
float computeColor(char color, float z);
void loadData(char* filename);
float scalePos(float x, float maxDim, float xMax);
float calculate_radius(float z, int r0, int rmin);

float3 *vel, *acc;
float4 *pos; 

__device__ float3 tile_calculation(float4 *d_pos, int myPos, int i, float3 acc)
{
	float distSqr, distSixth, force;
	float3 r;

	for(int j=i; j<blockDim.x; j++)
	{
		if( myPos == j)
			continue;

		r.x = (d_pos[myPos]).x - (d_pos[j]).x;
		r.y = (d_pos[myPos]).y - (d_pos[j]).y;
		r.z = (d_pos[myPos]).z - (d_pos[j]).z;

		r.x *=1000;
		r.y *=1000;
		r.z *=1000;

		distSqr = r.x*r.x + r.y*r.y + r.z*r.z + EPSI2;
		distSixth = distSqr*distSqr*distSqr;

		force = -(G*M*(d_pos[myPos]).w)/sqrtf(distSixth);

		acc.x += force*(r.x);
		acc.y += force*(r.y);
		acc.z += force*(r.z);
	}
	return acc;
}

__global__ void gravitational_forces(float4 *d_pos, float3 *vel, float3 *acc)
{
	int idx = ((blockDim.x)*(blockIdx.x)) + threadIdx.x;
	float3 newAcc = {0.0f, 0.0f, 0.0f};
	int tileSize = blockDim.x;
	int tile=0;

	(d_pos[idx]).x += TIME*( (vel[idx]).x + (TIME*(acc[idx]).x)/2 );
	(d_pos[idx]).y += TIME*( (vel[idx]).y + (TIME*(acc[idx]).y)/2 );
	(d_pos[idx]).z += TIME*( (vel[idx]).z + (TIME*(acc[idx]).z)/2 );

	for(int i=0; i<N; i+=tileSize, tile++)
	{
		newAcc = tile_calculation(d_pos, idx, i, newAcc);
	}

	(vel[idx]).x += TIME*((acc[idx]).x + newAcc.x)/2; 
	(vel[idx]).y += TIME*((acc[idx]).y + newAcc.y)/2;
	(vel[idx]).z += TIME*((acc[idx]).z + newAcc.z)/2;
	
	(acc[idx]).x = newAcc.x;
	(acc[idx]).y = newAcc.y;
	(acc[idx]).z = newAcc.z;
}

int main()
{
	Mat bg = imread("black bg.jpeg", 1);
	resize(bg, bg, Size(XDIM, YDIM));
	int i;
	char ch='a';
	char filename[] = NAME;

	pos = new float4[N];
	vel = new float3[N];
	acc = new float3[N];
	
	loadData(filename);

	float3 *d_vel, *d_acc;
	float4 *d_pos;

	cudaMalloc((void**)&d_pos, N*sizeof(float4));
	cudaMalloc((void**)&d_vel, N*sizeof(float3));
	cudaMalloc((void**)&d_acc, N*sizeof(float3));

	dim3 blockSize(512,1,1);
	dim3 gridSize(N/512,1,1);

	cudaMemcpy(d_pos, pos, N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, vel, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_acc, acc, N*sizeof(float3), cudaMemcpyHostToDevice);

	while(1)
	{
		ch = waitKey(TIME*1000);
		if(ch=='q' || ch=='Q' || ch==27)
			break;

		gravitational_forces<<< gridSize, blockSize >>>(d_pos, d_vel, d_acc);
		cudaDeviceSynchronize();

		cudaMemcpy(pos, d_pos, N*sizeof(float4), cudaMemcpyDeviceToHost);

		bg = imread("black bg.jpeg", 1);
		resize(bg, bg, Size(XDIM, YDIM));

		for(i=0; i<N; ++i)
		{
			draw(bg, pos[i], 2.8, 1.2);
		}

		resize(bg, bg, Size(800,800));

		imshow("N-Bodies", bg);
	}

	cudaFree(d_pos);
	cudaFree(d_vel);
	cudaFree(d_acc);
	delete pos;
	delete vel;
	delete acc;

	return 0;
}

void draw(Mat bg, float4 pos, int r0, int rmin)
{
	float radius = calculate_radius(pos.z, r0, rmin);

	circle(bg, Point((int)pos.x, (int)pos.y), radius, 
				Scalar(computeColor('b', pos.z), computeColor('g', pos.z), computeColor('r', pos.z)), -1, CV_AA);
	
	return;
}

float calculate_radius(float z, int r0, int rmin)
{
	return ((rmin-r0)*(z))/ZDIM + r0;
}

float computeColor(char color, float z)
{
	int fromColor, toColor;
	if( z > (ZDIM/2) )
	{
		switch(color)
		{
			case 'b' :
				fromColor = 247;
				toColor = 180;
				break;
			case 'g' :
				fromColor = 139;
				toColor = 135;
				break;
			case 'r' :
				fromColor = 111;
				toColor = 124;
				break;	
		}
		// float a = (toColor - fromColor)/(100 - ZDIM);
		// float b = fromColor - (100*a);
		float a = ((toColor - fromColor)*2)/ZDIM;
		return (a*z + fromColor);
	}
	else
	{
		switch(color)
		{
			case 'b' :
				return 180;
			case 'g' :
				return 135;
			case 'r' :
				return 124;
		}
	}
}

// function to load Data from a file
void loadData(char* filename)
{
    int bodies = N;
    int skip;

    // skip = 2;

    if( N <= 49152)
    	skip = 49152 / bodies;
    else
    	skip = 81920 / bodies;
    
    // total 81920 particles
	// 16384 Gal. Disk
	// 16384 And. Disk
	// 8192  Gal. bulge
	// 8192  And. bulge
	// 16384 Gal. halo
	// 16384 And. halo

    FILE *fin;
    
    if ((fin = fopen(filename, "r")))
    {
    
    	char buf[MAXSTR];
    	float v[7];
    	
    	int k=0;
    	for (int i=0; i< bodies; i++,k++)
    	{
    		// depend on input size...
    		for (int j=0; j < skip; j++,k++)
    			fgets (buf, MAXSTR, fin);	// lead line
    		
    		sscanf(buf, "%f %f %f %f %f %f %f", v+0, v+1, v+2, v+3, v+4, v+5, v+6);
    		
    		// position
    		(pos[i]).x = scalePos( v[1], XDIM, XMAX);
    		(pos[i]).y = scalePos( v[2], YDIM, YMAX);
    		(pos[i]).z = scalePos( v[3], ZDIM, ZMAX);
    		
    		// (pos[i]).x = v[1];
    		// (pos[i]).y = v[2];
    		// (pos[i]).z = v[3];

    		// mass
    		(pos[i]).w = v[0];

    		// velocity
    		(vel[i]).x = v[4]*velFactor;;
    		(vel[i]).y = v[5]*velFactor;;
    		(vel[i]).z = v[6]*velFactor;;

    		// acceleration
    		(acc[i]).x = (acc[i]).y = (acc[i]).z = 0;
    	}   
    }
    else
    {
    	printf("cannot find file...: %s\n", filename);
    	exit(0);
    }
}

float scalePos(float x, float maxDim, float xMax)
{
	float b = maxDim/2;
	float a = b/xMax;

	return ( a*x + b);
}