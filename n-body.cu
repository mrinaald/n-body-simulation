#include <iostream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define N 40960								// No. of particles in simulation.
#define NAME "galaxy.tab"					// Name of the dataset file.
#define TIME 0.02							// Time step between successive computation of kinametics of particles.
#define G 0.0000000000667					
#define M 10000000000000000					// Mass factor used to increase the value of force for calculation purposes.
#define EPSI2 0.0001						// Smallest error in calculation of distance
#define XDIM 2000							// Dimension in x-direction
#define YDIM 2000							// Dimension in y-direction
#define ZDIM 2000							// Dimension in z-direction
#define XMAX 38								// Maximum absolute value possible for x coordinates(checked from dataset)
#define YMAX 44 							// Maximum absolute value possible for y coordinates(checked from dataset)
#define ZMAX 38								// Maximum absolute value possible for z coordinates(checked from dataset)
#define velFactor 40.0f						
#define MAXSTR 256							

//----------------------------------------------------------------------------------------------------
// funtion to draw a circle at position 'pos'. 'r0' and 'rmin' are used for the computation of radius.
//----------------------------------------------------------------------------------------------------
void draw(Mat bg, float4 pos, int r0, int rmin);

//----------------------------------------------------------------------------------------------------
// function to compute the value of color relative to a value fixed at the reference level at ZDIM/2.
//----------------------------------------------------------------------------------------------------
float computeColor(char color, float z);

void loadData(char* filename);

float scalePos(float x, float maxDim, float xMax);

//----------------------------------------------------------------------------------------------------
// function to calculate the radius of particles using a linear function, 
// where 'r0' is the radius at z=ZDIM/2, and 'rmin' is the radius at z=(ZDIM-1)
//----------------------------------------------------------------------------------------------------
float calculate_radius(float z, int r0, int rmin);


float3 *vel, *acc;
float4 *pos; 


//-------------------------------------------------------------------------------------------
// device function to calculate acceleration of a tiled data,
// updating accelerations in each direction.
//------------------------------------------------------------------------------------------
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

		// increasing the distance in each dimension to make them of comparable values.
		r.x *=1000;
		r.y *=1000;
		r.z *=1000;

		distSqr = r.x*r.x + r.y*r.y + r.z*r.z + EPSI2;
		distSixth = distSqr*distSqr*distSqr;

		force = -(G*M*(d_pos[j]).w)/sqrtf(distSixth);

		acc.x += force*(r.x);
		acc.y += force*(r.y);
		acc.z += force*(r.z);
	}
	return acc;
}

//--------------------------------------------------------------------------
// kernal function to calculate the gravitational force for each particle.
// and updating velocities using "velocity Verlet method".
//--------------------------------------------------------------------------
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
	Mat bg = imread("space bg.jpeg", 1);
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

	// allocating memory to device pointers in device memory
	cudaMalloc((void**)&d_pos, N*sizeof(float4));
	cudaMalloc((void**)&d_vel, N*sizeof(float3));
	cudaMalloc((void**)&d_acc, N*sizeof(float3));

	//---------------------------------------------------
	// initializing of size of block in x dimension only
	// initializing of size of grid in x dimension only
	//---------------------------------------------------
	dim3 blockSize(512,1,1);				
	dim3 gridSize(N/512,1,1);

	cudaMemcpy(d_pos, pos, N*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, vel, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_acc, acc, N*sizeof(float3), cudaMemcpyHostToDevice);


//-------------------------------------------------------------------------------------------
// waitKey() function is used to introduce delay between successive cycles of calculation. 
// The argument to waitKey() is in milliseconds
// if user inputs 'q', 'Q' or 'ESC' key, the program ends.
//-------------------------------------------------------------------------------------------
	while(1)
	{
		ch = waitKey(TIME*1000);
		if(ch=='q' || ch=='Q' || ch==27)
			break;
		gravitational_forces<<< gridSize, blockSize >>>(d_pos, d_vel, d_acc);
		cudaDeviceSynchronize();
		cudaMemcpy(pos, d_pos, N*sizeof(float4), cudaMemcpyDeviceToHost);

		bg = imread("space bg.jpeg", 1);
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
//--------------------------------------------------------------------------------------
// function circle() used to draw particles at position (pos.x,pos.y) of radius 'radius'
//--------------------------------------------------------------------------------------
void draw(Mat bg, float4 pos, int r0, int rmin)
{
	float radius = calculate_radius(pos.z, r0, rmin);

	circle(bg, Point((int)pos.x, (int)pos.y), radius, 
				Scalar(computeColor('b', pos.z), computeColor('g', pos.z), computeColor('r', pos.z)), -1, CV_AA);
	
	return;
}
//-----------------------------------------------------------------------
// a linear function used to calculate radius of particle at pos.z
// 'r0' is the radius at z=ZDIM/2, and 'rmin' is the radius at z=(ZDIM-1)
//-----------------------------------------------------------------------
float calculate_radius(float z, int r0, int rmin)
{
	return ((rmin-r0)*(z))/ZDIM + r0;
}

//------------------------------------------------------
// linear function to calculate color of particle.
// fromColor is value of that color at z=0
// toColor is value of that color at z=(ZDIM/2)
//------------------------------------------------------
float computeColor(char color, float z)
{
	int fromColor, toColor;
	if( z < (ZDIM/2) )
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
//------------------------------------------------
// function to load Data from a file
//------------------------------------------------
void loadData(char* filename)
{
    int bodies = N;
    int skip;

    if( N <= 49152)
    	skip = 49152 / bodies;
    else
    	skip = 81920 / bodies;
    
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
    			fgets (buf, MAXSTR, fin);			// lead line
    		
    		sscanf(buf, "%f %f %f %f %f %f %f", v+0, v+1, v+2, v+3, v+4, v+5, v+6);
    		
    		// position		
       		(pos[i]).x = scalePos( v[1], XDIM, XMAX);
    		(pos[i]).y = scalePos( v[2], YDIM, YMAX);
    		(pos[i]).z = scalePos( v[3], ZDIM, ZMAX);
    		
       		// mass
    		(pos[i]).w = v[0];

    		// velocity
    		(vel[i]).x = v[4]*velFactor;;
    		(vel[i]).y = v[5]*velFactor;;
    		(vel[i]).z = v[6]*velFactor;;

    		(acc[i]).x = (acc[i]).y = (acc[i]).z = 0;
    	}   
    }
    else
    {
    	printf("cannot find file...: %s\n", filename);
    	exit(0);
    }
}
//-------------------------------------------------------------
//function to scale the position of particles to our dimensions
//----------------------------..........-----------------------
float scalePos(float x, float maxDim, float xMax)
{
	float b = maxDim/2;
	float a = b/xMax;

	return ( a*x + b);
}