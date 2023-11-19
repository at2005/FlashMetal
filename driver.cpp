

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include "rand.h"
#include <torch/torch.h>
#include <pybind11/pybind11.h>

#define CONVERT_MTL(input_tensor) ((MTL::Buffer*)(input_tensor.storage().data()))

std::string ReadMetalFile() {
	
        std::ifstream shader_file("flash.metal");
        std::stringstream text_buffer;
        text_buffer << shader_file.rdbuf();
	
	std::string temp_str = text_buffer.str();
        return temp_str; 
	

}



torch::Tensor& FlashMPSDispatch(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value, torch::Tensor& out) {
	
	// create metal device
	MTL::Device* dev = MTL::CreateSystemDefaultDevice();
	
	// print out GPU metadata
	std::cout << "MAX THREADGROUP MEMORY: " << dev->maxThreadgroupMemoryLength() << "\n";
	// create command queue where we will dispatch our jobs
	MTL::CommandQueue* cmd_queue = dev->newCommandQueue();

	std::string temp_file = ReadMetalFile();
	const char* str = temp_file.c_str();
	
	NS::Error* err = nullptr;
	MTL::Library* library= dev->newLibrary(NS::String::string(str, NS::UTF8StringEncoding), nullptr, &err);//newLibrary(NS::String::string("flash", NS::UTF8StringEncoding), nullptr, &err);

	if(library == nullptr) {
		 __builtin_printf( "%s", err->localizedDescription()->utf8String() );
		std::cout << "Error";

	}

	c10::IntArrayRef shape = query.sizes();
	unsigned int batch_size = shape[0];
	unsigned int num_heads = shape[1];
	unsigned int N_seq = shape[2];
	unsigned int n_embed = shape[3];


	// split into 16 blocks of size 4 each
	unsigned int Q_BLOCK_SIZE = 8; 
	unsigned int K_BLOCK_SIZE = 8;

	std::cout << "NUM_THREADS: " << (float)((float)N_seq / (float)Q_BLOCK_SIZE) << std::endl;
	std::cout << "VALUES_TO_COPY: " << (float)((float)(K_BLOCK_SIZE * K_BLOCK_SIZE * n_embed) / (float)N_seq) << std::endl;

	// PARAMETERS END
	// load function from metal shader file
	MTL::Function* kernelFunc = library->newFunction(NS::String::string("attention", NS::UTF8StringEncoding));
	MTL::ComputePipelineState* pipeline= dev->newComputePipelineState(kernelFunc, &err);
	

	
	// command queue and command buffer are where we send our jobs
	auto commandQueue = torch::mps::get_dispatch_queue();
	MTL::CommandBuffer* commandBuffer = (MTL::CommandBuffer*)(torch::mps::get_command_buffer());
	MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
	encoder->setComputePipelineState(pipeline);

	encoder->setBuffer(CONVERT_MTL(query), query.storage_offset(), 0);
	encoder->setBuffer(CONVERT_MTL(key), key.storage_offset(), 1);
	encoder->setBuffer(CONVERT_MTL(value), value.storage_offset(), 2);
	encoder->setBuffer(CONVERT_MTL(out), out.storage_offset(), 3); 
	

	// setting threads and threadgroup sizes
	MTL::Size threads_threadgroup; 
	MTL::Size threadgroup_per_grid;

	threads_threadgroup.height = N_seq / Q_BLOCK_SIZE;
	threads_threadgroup.width = 1;
	threads_threadgroup.depth = 1;

	threadgroup_per_grid.height = batch_size; 
	threadgroup_per_grid.width  = num_heads;
	threadgroup_per_grid.depth = 1;

	// perform computation	
	// dispatch threads to GPU
	encoder->dispatchThreadgroups(threadgroup_per_grid, threads_threadgroup);
	encoder->endEncoding();
	// commit jobs and wait before printing out
	torch::mps::commit();
	torch::mps::synchronize();

	std::cout << out << std::endl;
		
	dev->release();
	
	return out;
}

torch::Tensor FlashAttentionMPS(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value) {
	// PARAMETERS
	const unsigned int batch_size = 64;
	const unsigned int num_heads = 16;
	const unsigned int n_embed = 96;
	const unsigned int N_seq = 1024;

	int shape_arr[4] = {batch_size, num_heads, N_seq, n_embed};

	/*
	torch::Tensor query = torch::randn(shape_arr).to(torch::kMPS);
	torch::Tensor key = torch::randn(shape_arr).to(torch::kMPS);
	torch::Tensor value = torch::randn(shape_arr).to(torch::kMPS);
	*/

	// output tensor initialised to all zeros
	torch::Tensor out = torch::empty_like(value).to(torch::kMPS);
	
	return FlashMPSDispatch(query, key, value, out); 


}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("FlashAttentionMPS", &FlashAttentionMPS); 

}

