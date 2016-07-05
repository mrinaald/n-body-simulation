#include <iostream>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define N 8
#define TIME 0.05
#define G 0.0000000000667
#define M 10000000000000000
#define epsilon 0.00001

void draw(Mat bg, short x, short y);

__global__ void gravity(int *x, int *y, float *x_vel, float *y_vel, float *x_acc, float *y_acc)
{
	int idx = threadIdx.x;
	int i;
	double distance, force, new_xacc=0, new_yacc=0;
	
	x[idx] += (int)( TIME*x_vel[idx] + (TIME*TIME*x_acc[idx])/2);
	y[idx] += (int)( TIME*y_vel[idx] + (TIME*TIME*y_acc[idx])/2);

	for( i=0; i<N; ++i)
	{
		if( idx == i)
			continue;
		
		distance = epsilon + sqrt((double)( (x[idx]-x[i])*(x[idx]-x[i]) + (y[idx]-y[i])*(y[idx]-y[i])));
		// distance *= 0.001;

		force = -(G*M)/(distance*distance*distance);

		new_xacc += force*(x[idx]-x[i]);
		new_yacc += force*(y[idx]-y[i]);
	}
	// __syncthreads();

	// x[idx] = x[idx] + (int)( TIME*x_vel[idx] + (TIME*TIME*new_xacc)/2);
	// y[idx] = y[idx] + (int)( TIME*y_vel[idx] + (TIME*TIME*new_yacc)/2);
	// x[1] = y[1] = 100;
	// x[idx] += (int)( TIME*x_vel[idx] + (TIME*TIME*x_acc)/2);
	// y[idx] += (int)( TIME*y_vel[idx] + (TIME*TIME*y_acc)/2);

	x_vel[idx] += TIME*(x_acc[idx]+new_xacc)/2;
	y_vel[idx] += TIME*(y_acc[idx]+new_yacc)/2;
	x_acc[idx] = new_xacc;
	y_acc[idx] = new_yacc;

	return;
}

int main()
{
	Mat bg = imread("black bg.jpeg", 1);
	resize(bg, bg, Size(1300, 700));
	int i, j;
	bool flag=false;
	char ch='a';
	// cin >> N;
	// N=2;
	int *pos_x, *pos_y;
	float *x_vel, *y_vel;
	float *x_acc, *y_acc;

	pos_x = (int *)malloc(N*sizeof(int));
	pos_y = (int *)malloc(N*sizeof(int));
	x_vel = (float *)malloc(N*sizeof(float));
	y_vel = (float *)malloc(N*sizeof(float));
	x_acc = (float *)malloc(N*sizeof(float));
	y_acc = (float *)malloc(N*sizeof(float));

	srand( (unsigned)time( NULL ) );

	for( i=0; i<N; ++i)
	{
		pos_x[i] = ( (rand()%1225) + 12 );
		pos_y[i] = ( (rand()%675) + 12 );
		x_vel[i] = rand()%50;
		y_vel[i] = rand()%50;
		x_acc[i] = rand()%50;
		y_acc[i] = rand()%50;

		for(j=0; j<i; ++j)
		{
			if(pos_x[i] == pos_x[j] )
			{
				pos_x[i] = ( (rand()%1225) + 12 );
				flag = true;
			}
			if( pos_y[i] == pos_y[j] )
			{
				pos_y[i] = ( (rand()%675) + 12 );
				flag = true;
			}
			if(flag)
				j=-1;
		}
			
	}

	// pos_x[0] = pos_y[0] = 500;
	// pos_x[1] = 500;
	// pos_y[1] = 100;

	// x_vel[0] = -sqrt((G*M)/(4*200));
	// x_vel[1] = sqrt((G*M)/(4*200));
	// y_vel[0] = y_vel[1] = 0;
	// x_acc[0] = x_acc[1] = 0;
	// y_acc[0] = - ((x_vel[0])*(x_vel[0]))/400;
	// y_acc[1] = ((x_vel[0])*(x_vel[0]))/400;

	int *dev_x;
	int *dev_y;
	float *d_x_vel;
	float *d_y_vel;
	float *d_x_acc;
	float *d_y_acc;

	cudaMalloc((void**)&dev_x, N*sizeof(int));
	cudaMalloc((void**)&dev_y, N*sizeof(int));
	cudaMalloc((void**)&d_x_vel, N*sizeof(float));
	cudaMalloc((void**)&d_y_vel, N*sizeof(float));
	cudaMalloc((void**)&d_x_acc, N*sizeof(float));
	cudaMalloc((void**)&d_y_acc, N*sizeof(float));

	dim3 blockSize(N,1,1);
	dim3 gridSize(1,1,1);

	cudaMemcpy(d_x_acc, x_acc, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_acc, y_acc, N*sizeof(float), cudaMemcpyHostToDevice);

	while(1)
	{
		ch=waitKey(TIME*1000);
		// ch=waitKey(1000);
		if( (ch=='q') || (ch=='Q') || (ch==27) || (ch==' ') )
		{
			break;
		}
		
		for( i=0; i<N; ++i)
		{
			cout << pos_x[i] << ' ' << pos_y[i] << "\t\t" << x_vel[i] << '\t' << y_vel[i]<< endl;
		}

		cudaMemcpy(dev_x, pos_x, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y, pos_y, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_x_vel, x_vel, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y_vel, y_vel, N*sizeof(float), cudaMemcpyHostToDevice);
		
		gravity<<< gridSize, blockSize >>>(dev_x, dev_y, d_x_vel, d_y_vel, d_x_acc, d_y_acc);
		cudaDeviceSynchronize();

		cudaMemcpy(pos_x, dev_x, N*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(pos_y, dev_y, N*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(x_vel, d_x_vel, N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(y_vel, d_y_vel, N*sizeof(float), cudaMemcpyDeviceToHost);
		// cudaMemcpy(x_acc, d_x_acc, N*sizeof(float), cudaMemcpyDeviceToHost);
		// cudaMemcpy(y_acc, d_y_acc, N*sizeof(float), cudaMemcpyDeviceToHost);

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
	cudaFree(d_x_vel);
	cudaFree(d_y_vel);
	free(pos_x);
	free(pos_y);
	free(x_vel);
	free(y_vel);
	free(x_acc);
	free(y_acc);
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