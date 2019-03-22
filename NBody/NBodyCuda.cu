#include "main.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include "NBody.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
using namespace std;


static double time_stamp = 0.016;
static unsigned int currentRead = 0;
#define BLOCK_SIZE 256

double getRandom(double min, double max)
{
	double r = (double)rand() / RAND_MAX;
	return r*(max - min) + min;
}

/**
normalize the vector(vec_x,vec_y)
**/
void normalize(double & vec_x, double & vec_y)
{
	double r = sqrt(vec_x * vec_x + vec_y * vec_y);
	vec_x = r > 1e-10 ? vec_x : vec_x / r;
	vec_y = r > 1e-10 ? vec_y : vec_y / r;
}

void swap(int &a, int &b)
{
	int tmp = a;
	a = b;
	b = tmp;
}

double dot(double vec_a[], double vec_b[])
{
	return vec_a[0] * vec_b[0] * vec_a[1] * vec_b[1];
}

__device__ double2 bodyBodyInteraction(double2 ai,body bi, body bj,unsigned int currentRead)
{
	double2 r;
	r.x = bj.x[currentRead] - bi.x[currentRead];
	r.y = bj.y[currentRead] - bi.y[currentRead];
	double distSqrt = r.x * r.x + r.y * r.y + eps *eps;
	double distSixth = distSqrt * distSqrt * distSqrt;
	double invDistCube = 1.0 / sqrt(distSixth);
	double s = bj.m * invDistCube;
	ai.x += r.x * s;
	ai.y += r.y * s;
	return ai;
	return r;
}

__device__ double2 computeBodyAccel(body myBody,body * bodies, int numTiles,unsigned int currentRead)
{
	extern __shared__ body sharedBodies [];
	double2 accel = { 0.0,0.0 };
	for (int tile = 0; tile < numTiles; tile++)
	{
		/*int blockId = gridDim.x * blockIdx.y + numTiles;
		int threadId = blockId * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
		shareBodies[blockDim.x * threadIdx.y + threadIdx.x] = bodies[threadId];*/
		sharedBodies[threadIdx.x] = bodies[tile * blockDim.x + threadIdx.x];
		__syncthreads();
		for (unsigned int counter = 0; counter < blockDim.x; counter++)
		{
			accel = bodyBodyInteraction(accel, myBody, sharedBodies[counter],currentRead);
		}
		__syncthreads();
	}
	return accel;
}

//kernal function
__global__ void integrateBodies(body * __restrict__ bodies, unsigned int currentRead, double deltaTime, double dampings, int numTiles)
{
	/*int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y *blockDim.x) + threadIdx.x;
	int index = threadId;*/
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index > NUM_BODIES)
	{
		return;
	}
	body myBody = bodies[index];
	double2 accel = computeBodyAccel(myBody, bodies, numTiles, currentRead);
	myBody.vx += accel.x * deltaTime;
	myBody.vy += accel.y * deltaTime;
	double x = myBody.x[currentRead] + myBody.vx * deltaTime;
	double y = myBody.y[currentRead] + myBody.vy * deltaTime;
	if (x >= 1.0 || x <= -1.0)
	{
		if (x >= 1.0)
		{
		    x = 1 - (x - 1)* dampings / x;
			x = 0.5;
		}
		else
		{
			x = -1 + (x + 1)* dampings / -x;
			x = - 0.5;
		}
		myBody.vx *= -dampings;
	}
	if (y >= 1.0 || y <= -1.0)
	{
		if (y >= 1.0)
		{
			y = 1 - (y - 1)*dampings / y;
			y = 0.5;
		}
		else
		{
			y = -1 + (y + 1) * dampings / -y;
			y = -0.5;
		}
		myBody.vy *= -dampings;
	}
	myBody.x[1 - currentRead] = x;
	myBody.y[1 - currentRead] = y;	
	bodies[index] = myBody;
}

void integrateNBodySystem(body *bodies,double deltaTime, double dampings,unsigned int currentRead)
{
	/*dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid(NUM_BODIES / threadsPerBlock.x,NUM_BODIES / threadsPerBlock.y);
	int numTiles = NUM_BODIES / BLOCK_SIZE;*/
	int numBlocks = (NUM_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int numTiles = (NUM_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int sharedMemSize = BLOCK_SIZE * sizeof(body);
	integrateBodies << <numBlocks, BLOCK_SIZE,sharedMemSize>> >(
		bodies, currentRead,deltaTime, dampings,numTiles);
}

void rasterize(struct body* bodies, unsigned char* buffer)
{
	/**
	rasterize the bodies from x,y: (-1,-1) to (1,1) according to some kind of formula

	Note: You can change the code for better visualization
	As the following code can be parallelized, you can optimize this routine with CUDA.

	\param bodies A collection of bodies (located on the device).
	\param buffer the RGB buffer for screen display (located on the host).
	*/

	// clear the canvas
	size_t bytes = NUM_BODIES * sizeof(body);
	memset(buffer, 0, SCREEN_WIDTH*SCREEN_HEIGHT * 3 * sizeof(unsigned char));
	body *hostArray = (body *)malloc(bytes);
	cudaMemcpy(hostArray, bodies, bytes, cudaMemcpyDeviceToHost);
	//TODO: copy the device memory to the host, and draw points on the canvas

	// Following is a sample of drawing a nice picture to the buffer.
	// You will know the index for each pixel.
	// The pixel value is from 0-255 so the data type is in unsigned char.
	for (int i = 0; i < NUM_BODIES; i++)
	{
		try
		{
			double x_tmp = hostArray[i].x[currentRead];
			int x_screen = (int)(SCREEN_WIDTH * (x_tmp + 1.0) / 2.0);
			double y_tmp = hostArray[i].y[currentRead];
			int y_screen = (int)(SCREEN_HEIGHT * (y_tmp + 1.0) / 2.0);

			buffer[x_screen * SCREEN_WIDTH * 3 + y_screen * 3 + 0] = (unsigned char)255;//getRandom(0,255);
			buffer[x_screen * SCREEN_WIDTH * 3 + y_screen * 3 + 1] = (unsigned char)255;//getRandom(0, 255);
			buffer[x_screen * SCREEN_WIDTH * 3 + y_screen * 3 + 2] = (unsigned char)255;//getRandom(0, 255);
		}
		catch(exception e)
		{

		}
	}
	free(hostArray);
}

struct body* initializeNBodyCuda()
{
	/**
	initialize the bodies, then copy to the CUDA device memory
	return the device pointer so that it can be reused in the NBodyTimestepCuda function.
	*/

	// initialize the position and velocity
	// you can implement own initial conditions to form a sprial/ellipse galaxy, have fun.
	
	//This part is modified by Timmy Qiao on Oct.8.2018
	//Use getRandom function and get some random position and velocity
	body* bodies_h = new body[NUM_BODIES];
	for (int i = 0; i < NUM_BODIES; i++)
	{
		int currentRead = 0;
		double x, y,m;//both x and y range from [-1.0,1.0]
		x = getRandom(-1, 1);
		y = getRandom(-1, 1);
		bodies_h[i].x[0] = bodies_h[i].x[1] = x;
		bodies_h[i].y[0] = bodies_h[i].y[1] = y;

		x = y = 0.0;
		x = getRandom(-1, 1);
		y = getRandom(-1, 1);
		normalize(x, y);
		bodies_h[i].vx = x / 10.0;
		bodies_h[i].vy = y / 10.0;

		m = getRandom(0.001, 0.002);
		bodies_h[i].m = m;
	}
	body * bodies_d;
	cudaMalloc((void **)&bodies_d, NUM_BODIES * sizeof(body));
	cudaMemcpy(bodies_d, bodies_h, NUM_BODIES * sizeof(body), cudaMemcpyHostToDevice);
	free(bodies_h);
	return bodies_d;
}

void NBodyTimestepCuda(struct body* bodies, double rx, double ry, bool cursor)
{
	/**
	Compute a time step on the CUDA device.
	TODO: correctly manage the device memory, compute the time step with proper block/threads

	\param bodies A collection of bodies (located on the device).
	\param rx position x of the cursor.
	\param ry position y of the cursor.
	\param cursor Enable the mouse interaction if true (adding a weight = cursor_weight body in the computation).
	*/

	integrateNBodySystem(bodies,time_stamp, collision_damping,currentRead);
 	currentRead = 1 - currentRead;
}









