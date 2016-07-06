#include <iostream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define N 512
#define TIME 0.02
#define G 0.0000000000667
#define M 10000000000000000
#define EPSI2 0.0001
#define XDIM 700
#define YDIM 700
#define ZDIM 3500

void draw(Mat bg, float3 pos, int r0, int rmin);
float computeColor1(int radius, float maxRadius, float fromColor, float toColor);
float computeColor2(int radius, float maxRadius, float maxColor);
float computeColor3(int radius, float maxRadius, float maxColor);

__global__ void gravitational_force(float3 *pos, float3 *vel, float3 *acc)
{
	int idx = threadIdx.x;
	int i;
	float distSqr, distSixth, force;
	float3 newAcc;
	newAcc.x = newAcc.y = newAcc.z = 0;
	float3 r;

	(pos[idx]).x += TIME*( (vel[idx]).x + (TIME*(acc[idx]).x)/2 );
	(pos[idx]).y += TIME*( (vel[idx]).y + (TIME*(acc[idx]).y)/2 );
	(pos[idx]).z += TIME*( (vel[idx]).z + (TIME*(acc[idx]).z)/2 );

	for(i=0; i<N; ++i)
	{
		if(idx==i)
			continue;

		r.x = (pos[idx]).x - (pos[i]).x;
		r.y = (pos[idx]).y - (pos[i]).y;
		r.z = (pos[idx]).z - (pos[i]).z;

		r.x *=1000;
		r.y *=1000;
		r.z *=1000;

		distSqr = r.x*r.x + r.y*r.y + r.z*r.z + EPSI2;
		distSixth = distSqr*distSqr*distSqr;

		force = -(G*M)/sqrtf(distSixth);

		newAcc.x += force*(r.x);
		newAcc.y += force*(r.y);
		newAcc.z += force*(r.z);
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
	int i, j;
	char ch='a';
	bool flag=false;
	float3 *pos, *vel, *acc;
	
	pos = new float3[N];
	vel = new float3[N];
	acc = new float3[N];
	
	srand( (unsigned)time( NULL ) );

	for( i=0; i<N; ++i)
	{
		(pos[i]).x = (rand()%(XDIM-25));
		(pos[i]).y = (rand()%(YDIM-25));
		(pos[i]).z = (rand()%(ZDIM-25));
		(vel[i]).x = rand()%151 - 75;
		(vel[i]).y = rand()%151 - 75;
		(vel[i]).z = rand()%151 - 75;
		(acc[i]).x = rand()%11 - 5;
		(acc[i]).y = rand()%11 - 5;
		(acc[i]).z = rand()%11 - 5;

		// for(j=0; j<i; ++j)
		// {
		// 	if((pos[i]).x == (pos[j]).x )
		// 	{
		// 		(pos[i]).x = rand()%675;
		// 		flag = true;
		// 	}
		// 	if( (pos[i]).y == (pos[j]).y )
		// 	{
		// 		(pos[i]).y = rand()%675;
		// 		flag = true;
		// 	}
		// 	if( (pos[i]).z == (pos[j]).z )
		// 	{
		// 		(pos[i]).z = rand()%675;
		// 		flag = true;
		// 	}
		// 	if(flag)
		// 		j=-1;
		// }
	}

	float3 *d_pos, *d_vel, *d_acc;

	cudaMalloc((void**)&d_pos, N*sizeof(float3));
	cudaMalloc((void**)&d_vel, N*sizeof(float3));
	cudaMalloc((void**)&d_acc, N*sizeof(float3));

	dim3 blockSize(N,1,1);
	dim3 gridSize(1,1,1);

	cudaMemcpy(d_pos, pos, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, vel, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_acc, acc, N*sizeof(float3), cudaMemcpyHostToDevice);

	while(1)
	{
		ch = waitKey(TIME*1000);
		// ch = waitKey(0);
		if(ch=='q' || ch=='Q' || ch==27)
			break;

		for(i=0; i<N; ++i)
		{
			cout << i << ". (" << (pos[i]).x << ',' << (pos[i]).y << ',' << (pos[i]).z << ')' << endl;
		}

		cout << endl;

		gravitational_force<<< gridSize, blockSize >>>(d_pos, d_vel, d_acc);
		cudaDeviceSynchronize();

		cudaMemcpy(pos, d_pos, N*sizeof(float3), cudaMemcpyDeviceToHost);

		bg = imread("black bg.jpeg", 1);
		resize(bg, bg, Size(XDIM, YDIM));

		for(i=0; i<N; ++i)
		{
			draw(bg, pos[i], 10, 1);
		}

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

void draw(Mat bg, float3 pos, int r0, int rmin)
{
	if( pos.x < 0-40 || pos.x > XDIM+40 || pos.y < 0-40 || pos.y > YDIM+40)
		return;

	float radius = ((rmin-r0)*(pos.z))/ZDIM + r0;

	// circle(bg,Point((int)pos.x,(int)pos.y),radius,Scalar(255,0,0),-1, CV_AA);

	for(int i=0; i<=radius; ++i)
	{
		// if( pos.z > 350 )
		// {
			circle(bg,Point((int)pos.x,(int)pos.y),i,
				 Scalar(computeColor1(i, radius, 47, 1),computeColor1(i, radius, 171, 177),computeColor1(i, radius, 224, 252)),2, CV_AA);
		// }
		// else
		// {
			// circle(bg,Point((int)pos.x,(int)pos.y),i,Scalar(0,computeColor3(i, radius, 221),computeColor3(i, radius, 255)),2, CV_AA);
		// }
	}
	
	// radius = 15;
	// for(int i=1; i<=radius; i+=1)
	// 	circle(bg,Point(100,100),i,Scalar(computeColor1(i, radius, 225),computeColor1(i, radius, 225),computeColor1(i, radius, 255)),1, CV_AA);

	// for(int i=1; i<=radius; i+=1)
	// 	circle(bg,Point(200,200),i,Scalar(0,computeColor2(i, radius, 221),computeColor2(i, radius, 255)),1, CV_AA);

	// for(int i=1; i<=radius; i+=1)
	// 	circle(bg,Point(300,300),i,Scalar(0,computeColor3(i, radius, 221),computeColor3(i, radius, 255)),1, CV_AA);

	// radius = 10;
	// for(int i=1; i<=radius; i+=1)
	// 	circle(bg,Point(400,400),i,Scalar(0,computeColor1(i, radius, 221),computeColor1(i, radius, 255)),1, CV_AA);

	return;
}

// using a linear function
float computeColor1(int radius, float maxRadius, float fromColor, float toColor)
{
	// float a,b,c;
	
	// b = 1 - maxRadius;
	// a = maxColor/b;
	// c = -(maxColor*maxRadius)/b;

	float a = (toColor - fromColor)/(maxRadius-1);
	float b = fromColor - a;

	return ( a*radius + b);		
}

// using a quadratic function
float computeColor2(int radius, float maxRadius, float maxColor)
{
	float R2 = maxRadius*maxRadius;
	float b = 1 - R2;
	float a = maxColor/b;
	float c = -(maxColor*R2)/b;

	return ( (a*radius*radius) + c);
}

// using a cubic function
float computeColor3(int radius, float maxRadius, float maxColor)
{
	float R3 = maxRadius*maxRadius*maxRadius;
	float b = 1-R3;
	float a = maxColor/b;
	float c = -(R3*maxColor)/b;

	return ( (a*radius*radius*radius) + c);
}