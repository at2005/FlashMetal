

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>

int main() {
	MTL::Device* dev = MTL::CreateSystemDefaultDevice();
	//std::cout << dev->maxThreadgroupMemoryLength() << "\n";
	MTL::CommandQueue* cmd_queue = dev->newCommandQueue();
	
	int mat_size = 50;
	int total_el_size = mat_size * mat_size;
	
	float buffer_cpu[total_el_size]; 

	for(int i = 0; i < total_el_size; i++) {
		buffer_cpu[i] = i;
	}


	const char* str = R"(
		

kernel void mat_mul(const device float* mat1 [[buffer(0)]], const device float* mat2 [[buffer(1)]], device float* out [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
	float total = 0;
	int i = gid.y;
	int j = gid.x;
	int ROW_SIZE = 50;
	for(int k = 0; k < ROW_SIZE; k++) {
		total += mat1[ROW_SIZE * k + i] * mat2[j* ROW_SIZE + k];
	}

	out[j*ROW_SIZE + i] = total;

}
	)";

	NS::Error* err = nullptr;
	MTL::Library* library= dev->newLibrary(NS::String::string(str, NS::UTF8StringEncoding), nullptr, &err);//newLibrary(NS::String::string("flash", NS::UTF8StringEncoding), nullptr, &err);

	if(library == nullptr) {
		 __builtin_printf( "%s", err->localizedDescription()->utf8String() );
		std::cout << "Error";

	}


	MTL::Function* kernelFunc = library->newFunction(NS::String::string("mat_mul", NS::UTF8StringEncoding));
	MTL::ComputePipelineState* pipeline= dev->newComputePipelineState(kernelFunc, &err);
	MTL::Buffer* buff1 = dev->newBuffer(total_el_size, MTL::ResourceStorageModeShared);
	MTL::Buffer* buff2 = dev->newBuffer(total_el_size, MTL::ResourceStorageModeShared);
	MTL::Buffer* buff_out = dev->newBuffer(total_el_size, MTL::ResourceStorageModeShared);
	
	MTL::CommandQueue* commandQueue = dev->newCommandQueue();
	MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
	encoder->setComputePipelineState(pipeline);

	memcpy(buff1->contents(), buffer_cpu, total_el_size * sizeof(float));
	memcpy(buff2->contents(), buffer_cpu, total_el_size * sizeof(float));

	encoder->setBuffer(buff1, 0, 0);
	encoder->setBuffer(buff2, 0, 1);
	encoder->setBuffer(buff_out, 0, 2);

	MTL::Size threads_threadgroup; 
	MTL::Size threadgroup_per_grid;

	threads_threadgroup.height = 1;
	threads_threadgroup.width = 1;
	threads_threadgroup.depth = 1;

	threadgroup_per_grid.height = 50;
	threadgroup_per_grid.width  = 50;
	threadgroup_per_grid.depth = 1;

	
	encoder->dispatchThreadgroups(threadgroup_per_grid, threads_threadgroup);
	encoder->endEncoding();
	commandBuffer->commit();
	commandBuffer->waitUntilCompleted();
	float* output_buffer = (float*)(buff_out->contents());
	

	for(int i = 0; i < total_el_size; i++) {
		std::cout << output_buffer[i] << "\n";
	}

	dev->release();


}



