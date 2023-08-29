

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>

int main() {
	MTL::Device* dev = MTL::CreateSystemDefaultDevice();
	std::cout << dev->maxThreadgroupMemoryLength() << "\n";
	MTL::CommandQueue* cmd_queue = dev->newCommandQueue();
	
	dev->release();


}



