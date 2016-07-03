#include <iostream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define N 2
#define TIME 0.1
#define G 0.0000000000667
#define M 10000000000000000
#define epsilon 0.00001

void draw(Mat bg, short x, short y);

__global__ void gravity(int *x, int *y, float *vel_x, float *vel_y)
{
	int idx = threadIdx.x;
	int i;
	double distance, force, acc_x=0, acc_y=0;
	for( i=0; i<N; ++i)
	{
		if( idx == i)
			continue;
		
		distance = epsilon + sqrt((double)( (x[idx]-x[i])*(x[idx]-x[i]) + (y[idx]-y[i])*(y[idx]-y[i])));
		// distance *= 0.001;

		force = -(G*M)/(distance*distance*distance);

		acc_x += force*(x[idx]-x[i]);
		acc_y += force*(y[idx]-y[i]);
	}
	// __syncthreads();

	// x[idx] = x[idx] + (int)( TIME*vel_x[idx] + (TIME*TIME*acc_x)/2);
	// y[idx] = y[idx] + (int)( TIME*vel_y[idx] + (TIME*TIME*acc_y)/2);
	// x[1] = y[1] = 100;
	x[idx] += (int)( TIME*vel_x[idx] + (TIME*TIME*acc_x)/2);
	y[idx] += (int)( TIME*vel_y[idx] + (TIME*TIME*acc_y)/2);

	vel_x[idx] += TIME*acc_x;
	vel_y[idx] += TIME*acc_y;
	return;
}

int main()
{
	Mat bg = imread("black bg.jpeg", 1);
	resize(bg, bg, Size(1300, 700));
	int i, j, ch='a';
	bool flag=false;
	// cin >> N;
	// N=2;
	int *pos_x, *pos_y;
	float *vel_x, *vel_y;

	pos_x = (int *)malloc(N*sizeof(int));
	pos_y = (int *)malloc(N*sizeof(int));
	vel_x = (float *)malloc(N*sizeof(float));
	vel_y = (float *)malloc(N*sizeof(float));

	srand( (unsigned)time( NULL ) );

	// for( i=0; i<N; ++i)
	// {
	// 	pos_x[i] = ( (rand()%1225) + 12 );
	// 	pos_y[i] = ( (rand()%675) + 12 );
	// 	for(j=0; j<i; ++j)
	// 	{
	// 		if(pos_x[i] == pos_x[j] )
	// 		{
	// 			pos_x[i] = ( (rand()%1225) + 12 );
	// 			flag = true;
	// 		}
	// 		if( pos_y[i] == pos_y[j] )
	// 		{
	// 			pos_y[i] = ( (rand()%675) + 12 );
	// 			flag = true;
	// 		}
	// 		if(flag)
	// 			j=-1;
	// 	}
			
	// }

	pos_x[0] = pos_y[0] = 500;
	pos_x[1] = 500;
	pos_y[1] = 100;

	vel_x[0] = -sqrt((G*M)/(4*200));
	vel_x[1] = sqrt((G*M)/(4*200));
	vel_y[0] = vel_y[1] = 0;

	int *dev_x;
	int *dev_y;
	float *d_vel_x;
	float *d_vel_y;

	cudaMalloc((void**)&dev_x, N*sizeof(int));
	cudaMalloc((void**)&dev_y, N*sizeof(int));
	cudaMalloc((void**)&d_vel_x, N*sizeof(float));
	cudaMalloc((void**)&d_vel_y, N*sizeof(float));

	dim3 blockSize(N,1,1);
	dim3 gridSize(1,1,1);

	while(1)
	{
		ch=waitKey(TIME*1000);
		// ch=waitKey(1000);
		if( (ch=='q') || (ch=='Q') || (ch==27) || (ch==' ') )
			break;

		for( i=0; i<N; ++i)
		{
			cout << pos_x[i] << ' ' << pos_y[i] << "\t\t" << vel_x[i] << '\t' << vel_y[i]<< endl;
		}

		cudaMemcpy(dev_x, pos_x, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y, pos_y, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vel_x, vel_x, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vel_y, vel_y, N*sizeof(float), cudaMemcpyHostToDevice);

		gravity<<< gridSize, blockSize >>>(dev_x, dev_y, d_vel_x, d_vel_y);
		cudaDeviceSynchronize();

		cudaMemcpy(pos_x, dev_x, N*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(pos_y, dev_y, N*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(vel_x, d_vel_x, N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(vel_y, d_vel_y, N*sizeof(float), cudaMemcpyDeviceToHost);

		bg = imread("black bg.jpeg", 1);
		resize(bg, bg, Size(1300, 700));
		
		for( i=0; i<N; ++i)
		{
			draw(bg, pos_x[i], pos_y[i]);
		}

		imshow("N-Bodies", bg);
		
		// ch=waitKey(TIME*1000);
		// ch=waitKey(1000);
		// if( (ch=='q') || (ch=='Q') || (ch==27) )
		// 	break;
	}

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(d_vel_x);
	cudaFree(d_vel_y);
	return 0;
}

void draw(Mat bg, short x, short y)
{
	circle(bg,Point(x,y),10,Scalar(255,0,0),-1);
	circle(bg,Point(x,y),9,Scalar(255,32,32),-1);
	circle(bg,Point(x,y),8,Scalar(255,64,64),-1);
	circle(bg,Point(x,y),7,Scalar(255,96,96),-1);
	circle(bg,Point(x,y),6,Scalar(255,128,128),-1);
	circle(bg,Point(x,y),5,Scalar(255,160,160),-1);
	circle(bg,Point(x,y),4,Scalar(255,192,192),-1);
	circle(bg,Point(x,y),3,Scalar(255,224,224),-1);
	circle(bg,Point(x,y),2,Scalar(255,255,255),-1);
	return;
}