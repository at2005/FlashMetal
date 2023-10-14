
#include "rand.h"
#include <math.h>

CustomRandom::CustomRandom(long seed) {
	this->seed = seed;
	this->a = 1664525;
	this->c = 1013904223;
	this->m = pow(2, 32);
	this->state = seed;
} 


float CustomRandom::generate() {
	this->state = (this->a * this->state + this->c) % this->m;
	return (float)(this->state) / (float)(this->m);

}
