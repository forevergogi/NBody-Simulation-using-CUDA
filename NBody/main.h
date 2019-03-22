#ifndef SAKUYA_H
#define SAKUYA_H
#include<vector_types.h>

//Screen dimension constants
const int SCREEN_WIDTH = 480;
const int SCREEN_HEIGHT = 480;

// numbers of bodies in the simulation 
const int NUM_BODIES = 8192;

// the body structure
struct body
{
	double x[2];	// the x position
	double y[2];	// the y position
	double vx;	// the x-axis velocity
	double vy;	// the y-axis velocity
	double m;	// the body mass
};


struct Test
{
	double x;
    double y;
};

// function definitions for the main procedure
void rasterize(struct body* bodies, unsigned char* buffer);
struct body* initializeNBodyCuda();
void NBodyTimestepCuda(struct body* bodies, double rx, double ry, bool cursor);
#endif