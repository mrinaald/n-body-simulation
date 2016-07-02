#include <iostream>
#include <ctime>
#include <cstdlib>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void draw(Mat bg, short x, short y);

int main()
{
	Mat bg = imread("black bg.jpeg", 1);
	resize(bg, bg, Size(1300, 700));
	int N, i, j;
	bool flag=false;
	cin >> N;

	int *x, *y;

	x = (int *)malloc(N*sizeof(int));
	y = (int *)malloc(N*sizeof(int));

	srand( (unsigned)time( NULL ) );

	for( i=0; i<N; ++i)
	{
		x[i] = ( (rand()%1225) + 12 );
		y[i] = ( (rand()%675) + 12 );
		for(j=0; j<i; ++j)
		{
			if(x[i] == x[j] )
			{
				x[i] = ( (rand()%1225) + 12 );
				flag = true;
			}
			if( y[i] == y[j] )
			{
				y[i] = ( (rand()%675) + 12 );
				flag = true;
			}
			if(flag)
				j=-1;
		}
			
	}

	for( i=0; i<N; ++i)
	{
		cout << x[i] <<" " << y[i] << endl;
	}

	for( i=0; i<N; ++i)
	{
		draw(bg, x[i], y[i]);
	}

	imshow("N-Bodies", bg);
	waitKey(0);

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