
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

#define CONVERT_MTL(input_tensor) ((MTL::Buffer*)(input_tensor.storage().data()))

torch::Tensor& FlashMPSDispatch(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value, torch::Tensor& out, torch::Tensor& dO, torch::Tensor& out_dV, torch::Tensor& row_sums, torch::Tensor& row_max_vals) {
	
	// create metal device
	MTL::Device* dev = MTL::CreateSystemDefaultDevice();
	// print out GPU metadata
	// create command queue where we will dispatch our jobs
	MTL::CommandQueue* cmd_queue = dev->newCommandQueue();
	
	NS::Error* err = nullptr;
	NS::String* filePath = NS::String::alloc()->string("flashback.metallib", NS::StringEncoding::ASCIIStringEncoding);
	
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

	std::cout << "NUM_THREADS: " << (float)((float)N_seq / (float)Q_BLOCK_SIZE) << std::endl;
	std::cout << "VALUES_TO_COPY: " << (float)((float)(K_BLOCK_SIZE * K_BLOCK_SIZE * n_embed) / (float)N_seq) << std::endl;

	// PARAMETERS END
	// load function from metal shader file
	MTL::Function* kernelFunc = library->newFunction(NS::String::string("backprop_attention", NS::UTF8StringEncoding));
	std::cout << kernelFunc << std::endl;
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
	encoder->setBuffer(CONVERT_MTL(dO), dO.storage_offset(), 4); 
	encoder->setBuffer(CONVERT_MTL(out_dV), out_dV.storage_offset(), 5); 
	encoder->setBuffer(CONVERT_MTL(row_sums), row_sums.storage_offset(), 6); 
	encoder->setBuffer(CONVERT_MTL(row_max_vals), row_max_vals.storage_offset(), 7); 


	// setting threads and threadgroup sizes
	MTL::Size threads_threadgroup; 
	MTL::Size threadgroup_per_grid;

	threads_threadgroup.height = N_seq / Q_BLOCK_SIZE;
	std::cout << N_seq / Q_BLOCK_SIZE;
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
	
	return out_dV;
}

torch::Tensor FlashAttentionMPS(torch::Tensor& query, torch::Tensor& key, torch::Tensor& value, torch::Tensor& dO, torch::Tensor& out_dV) {
	// PARAMETERS
	const unsigned int batch_size = 1;
	const unsigned int num_heads = 1;
	const unsigned int n_embed = 96;
	const unsigned int N_seq = 1024;
	
	auto attn_scores = torch::matmul(query, key.transpose(-1, -2)) / std::sqrt(n_embed);

    	auto max_value_tuple = torch::max(attn_scores, -1);
	torch::Tensor max_values = std::get<0>(max_value_tuple); 
    	//std::cout << max_values << std::endl;	
	max_values = max_values.unsqueeze(-1);	
	auto exp_attn = torch::exp(attn_scores - max_values);
	auto row_sums = torch::sum(exp_attn, -1);
//	std::cout << row_sums.sizes() << max_values.sizes()<< std::endl;

 //   	auto attn_probs = torch::softmax(attn_scores, -1);
	
//	auto naive_dV = torch::matmul(attn_probs.transpose(-1, -2), dO);
//	std::cout << naive_dV;	
	// output tensor initialised to all zeros
	torch::Tensor out = torch::zeros({1,1,1024,1024}).to(torch::kMPS);
//	std::cout << exp_attn << std::endl << std::endl;
//	std::cout << torch::sum(attn_probs, -1) << std::endl;
//	std::cout << naive_dV << "\n\n\n\n";
	return  FlashMPSDispatch(query, key, value, out, dO, out_dV, row_sums, max_values);


}

int main() {
	torch::Tensor query = torch::randn({1, 1, 1024, 96}).to(torch::kMPS); 
	torch::Tensor key = torch::randn({1, 1, 1024, 96}).to(torch::kMPS); 
	torch::Tensor value = torch::randn({1, 1, 1024, 96}).to(torch::kMPS); 
	torch::Tensor dO = torch::randn({1,1, 1024, 96}).to(torch::kMPS);
	torch::Tensor out_dV = torch::zeros({1,1,1024, 96}).to(torch::kMPS);
	
//	std::cout << out_dV << std::endl;

	std::cout << (FlashAttentionMPS(query, key, value, dO, out_dV)); 
//FlashAttentionMPS(query, key, value, dO, out_dV); 

	return 0;
	

}


	
