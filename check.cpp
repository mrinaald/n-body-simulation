#include <iostream>
#include <limits>
#include <fstream>

using namespace std;

#define MAXSTR 256
#define N 81920

float max(float arr[]);
float min(float arr[]);

int main()
{
	FILE *fin;
    float *x, *y, *z, *w, *vx, *vy, *vz;

    if ((fin = fopen("dubinski.tab", "r")))
    {
    
    	char buf[MAXSTR];
    	float v[7];
    	// int idx = 0;
    	
    	// allocate memory
    	// gPos	= (float*)malloc(sizeof(float)*bodies*4);
    	// gVel	= (float*)malloc(sizeof(float)*bodies*4);
 	   	x = new float[N];
	    y = new float[N];
	    z = new float[N];
	    vx = new float[N];
	    vy = new float[N];
	    vz = new float[N];
	    w = new float[N];

    	// total 81920 particles
    	// 16384 Gal. Disk
    	// 16384 And. Disk
    	// 8192  Gal. bulge
    	// 8192  And. bulge
    	// 16384 Gal. halo
    	// 16384 And. halo
    	
    	int k=0;
    	for (int i=0; i< N; i++,k++)
    	{
    		// depend on input size...
    		// for (int j=0; j < skip; j++,k++)
    		fgets (buf, MAXSTR, fin);	// lead line
    		
    		sscanf(buf, "%f %f %f %f %f %f %f", v+0, v+1, v+2, v+3, v+4, v+5, v+6);
    		
    		// update index
    		// idx = i * 4;
    		
    		// // position
    		// gPos[idx+0] = v[1]*scaleFactor;
    		// gPos[idx+1] = v[2]*scaleFactor;
    		// gPos[idx+2] = v[3]*scaleFactor;
    		
    		// // mass
    		// gPos[idx+3] = v[0]*massFactor;
    		// //gPos[idx+3] = 1.0f;
    		// //printf("mass : %f\n", gPos[idx+3]);
    		
    		// // velocity
    		// gVel[idx+0] = v[4]*velFactor;
    		// gVel[idx+1] = v[5]*velFactor;
    		// gVel[idx+2] = v[6]*velFactor;
    		// gVel[idx+3] = 1.0f;
    		
    		x[i]= v[1];
    		y[i]=v[2];
    		z[i]=v[3];
    		w[i]=v[0];
    		vx[i]=v[4];
    		vy[i]=v[5];
    		vz[i]=v[6];
    	}

    	cout << "maxMass = " << max(w) << endl;
    	cout << "minMass = " << min(w) << endl << endl;
    	cout << "maxX = " << max(x) << endl;
    	cout << "minX = " << min(x) << endl << endl;
    	cout << "maxY = " << max(y) << endl;
    	cout << "minY = " << min(y) << endl << endl;
    	cout << "maxZ = " << max(z) << endl;
    	cout << "minZ = " << min(z) << endl << endl;
    	cout << "maxVX = " << max(vx) << endl;
    	cout << "minVX = " << min(vx) << endl << endl;
    	cout << "maxVY = " << max(vy) << endl;
    	cout << "minVY = " << min(vy) << endl << endl;
    	cout << "maxVZ = " << max(vz) << endl;
    	cout << "minVZ = " << min(vz) << endl << endl;
    }
    else
    {
    	printf("cannot find file...: \n");
    	return 0;
    }
	return 0;
}

float max(float arr[])
{
	// float maxValue = FLT_MIN;
	float maxValue = numeric_limits<float>::min();

	for(int i=0; i<N; ++i)
	{
		if( arr[i] > maxValue )
			maxValue = arr[i];
	}
	return maxValue;
}

float min(float arr[])
{
	float minValue = numeric_limits<float>::max();

	for(int i=0; i<N; ++i)
	{
		if( arr[i] < minValue )
			minValue = arr[i];
	}
	return minValue;
}