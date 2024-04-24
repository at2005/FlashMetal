

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
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#define CONVERT_MTL(input_tensor) ((MTL::Buffer*)(input_tensor.storage().data()))

std::string ReadMetalFile() {
	
        std::ifstream shader_file("flash.metal");
        std::stringstream text_buffer;
        text_buffer << shader_file.rdbuf();
	
	std::string temp_str = text_buffer.str();
        return temp_str; 
	

}

torch::Tensor& FlashMPSDispatch(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value, torch::Tensor& out, torch::Tensor& row_max, torch::Tensor& row_sum) {
	
	// create metal device
	MTL::Device* dev = MTL::CreateSystemDefaultDevice();
	
	// print out GPU metadata
	// create command queue where we will dispatch our jobs
	MTL::CommandQueue* cmd_queue = dev->newCommandQueue();
	
	NS::Error* err = nullptr;
	NS::String* filePath = NS::String::alloc()->string("flash.metallib", NS::StringEncoding::ASCIIStringEncoding);
	MTL::Library* library = dev->newLibrary(filePath, NULL);	

	if(library == nullptr) {
		 __builtin_printf( "%s", err->localizedDescription()->utf8String() );

	}

	c10::IntArrayRef shape = query.sizes();
	unsigned int batch_size = shape[0];
	unsigned int num_heads = shape[1];
	unsigned int N_seq = shape[2];
	unsigned int n_embed = shape[3];


	// split into 16 blocks of size 4 each
	unsigned int Q_BLOCK_SIZE = 8; 
	unsigned int K_BLOCK_SIZE = 8;

//	std::cout << "NUM_THREADS: " << (float)((float)N_seq / (float)Q_BLOCK_SIZE) << std::endl;
//	std::cout << "VALUES_TO_COPY: " << (float)((float)(K_BLOCK_SIZE * K_BLOCK_SIZE * n_embed) / (float)N_seq) << std::endl;

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
	encoder->setBuffer(CONVERT_MTL(row_max), out.storage_offset(), 4); 
	encoder->setBuffer(CONVERT_MTL(row_sum), out.storage_offset(), 5); 
	

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
		
	dev->release();
	
	return out;
}

torch::Tensor FlashAttentionMPS(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value) {
	// PARAMETERS
	const unsigned int batch_size = 64;
	const unsigned int num_heads = 16;
	const unsigned int n_embed = 96;
	const unsigned int N_seq = 1024;

	// output tensor initialised to all zeros
	torch::Tensor out = torch::empty_like(value).to(torch::kMPS);
	torch::Tensor row_max = torch::empty({batch_size, num_heads, N_seq});
	torch::Tensor row_sum = torch::empty({batch_size, num_heads, N_seq});
	
	return FlashMPSDispatch(query, key, value, out, row_max, row_sum); 


}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("FlashAttentionMPS", &FlashAttentionMPS); 

}

