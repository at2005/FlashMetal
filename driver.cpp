

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

std::string ReadMetalFile() {
	
        std::ifstream shader_file("flash.metal");
        std::stringstream text_buffer;
        text_buffer << shader_file.rdbuf();
	
	std::string temp_str = text_buffer.str();
        return temp_str; 
	

}


void create_prod_arr(int* shape_arr, int arr_size, int* output_arr, int curr_index) {
	if(curr_index == arr_size-1) {
		output_arr[arr_size - 1] = shape_arr[arr_size-1];	
	}

	else {
		output_arr[curr_index] = output_arr[curr_index+1] * shape_arr[curr_index];
	}	

	if(curr_index == 0) return;

	create_prod_arr(shape_arr, arr_size, output_arr, curr_index-1);
}




void print_tensor(float* buff, int* shape, unsigned int num_dim) {
	// strategy: print in reverse dimensions
	
	int prod_arr[num_dim];
	create_prod_arr(shape, num_dim, prod_arr, num_dim-1);

	int total_elements = prod_arr[0];

	for(int j = 0; j < total_elements; j++) { 
		printf("%f ", buff[j]); 
		for(int i = 0; i < num_dim; i++) {
			if((j+1) % prod_arr[i] == 0) {
				printf("\n");
			}

		}
	}


}


int main() {
	
	// create metal device
	MTL::Device* dev = MTL::CreateSystemDefaultDevice();
	
	// print out GPU metadata
	std::cout << "MAX THREADGROUP MEMORY: " << dev->maxThreadgroupMemoryLength() << "\n";
//	std::cout << dev->recommendedMaxWorkingSetSize() << "\n";
	// create command queue where we will dispatch our jobs
	MTL::CommandQueue* cmd_queue = dev->newCommandQueue();

	std::string temp_file = ReadMetalFile();
	const char* str = temp_file.c_str();

//	std::cout << str << "\n";
	
	// copy-pasted error-handling code, too bored to write it :)
	NS::Error* err = nullptr;
	MTL::Library* library= dev->newLibrary(NS::String::string(str, NS::UTF8StringEncoding), nullptr, &err);//newLibrary(NS::String::string("flash", NS::UTF8StringEncoding), nullptr, &err);

	if(library == nullptr) {
		 __builtin_printf( "%s", err->localizedDescription()->utf8String() );
		std::cout << "Error";

	}


/*

	unsigned int n_embed = 6;
	unsigned int N_seq = 6;
	int total_el_size = N_seq * n_embed; 

	// load function from metal shader file
	MTL::Function* kernelFunc = library->newFunction(NS::String::string("mat_mul", NS::UTF8StringEncoding));
	MTL::ComputePipelineState* pipeline= dev->newComputePipelineState(kernelFunc, &err);


	// attention buffers, ie Q,K,V, shape = (N_seq, n_embed)
	MTL::Buffer* query= dev->newBuffer(total_el_size, MTL::ResourceStorageModeShared);
	MTL::Buffer* key= dev->newBuffer(total_el_size, MTL::ResourceStorageModeShared);
	MTL::Buffer* value= dev->newBuffer(total_el_size, MTL::ResourceStorageModeShared);
	
	// ancillary buffers, shape = (N_seq)
	MTL::Buffer* l_vals =  dev->newBuffer(N_seq, MTL::ResourceStorageModeShared);
	MTL::Buffer* m_vals =  dev->newBuffer(N_seq, MTL::ResourceStorageModeShared); 

	// Output, shape = (N_seq, n_embed)	
	MTL::Buffer* buff_out = dev->newBuffer(total_el_size, MTL::ResourceStorageModeShared);
	


	
	// copying data into CPU buffer
	float buffer_cpu[total_el_size]; 

	for(int i = 0; i < total_el_size; i++) {
		buffer_cpu[i] = i;
		(float*)(buff_out->contents())[i] = 2.5f;
	}
	

	// copying CPU buffers into HBM memory buffer objects
	memcpy(query->contents(), buffer_cpu, total_el_size * sizeof(float));
	memcpy(key->contents(), buffer_cpu, total_el_size * sizeof(float));
	memcpy(value->contents(), buffer_cpu, total_el_size * sizeof(float));

	// copying -inf to m_vals and 0 to l_vals:
	for(int i = 0; i < N_seq; i++) {
		(float*)(l_vals->contents())[i] = 0;
		(float*)(m_vals->contents())[i] = -INFINITY;

	}	


	MTL::CommandQueue* commandQueue = dev->newCommandQueue();
	MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
	

	encoder->setComputePipelineState(pipeline);


	encoder->setBuffer(query, 0, 0);
	encoder->setBuffer(key, 0, 1);
	encoder->setBuffer(value, 0, 2);
	encoder->setBuffer(buff_out, 0, 3);
	encoder->setBuffer(l_vals, 0, 4);
	encoder->setBuffer(m_vals, 0, 5);

	// setting threads and threadgroup sizes
	MTL::Size threads_threadgroup; 
	MTL::Size threadgroup_per_grid;

	threads_threadgroup.height = 1;
	threads_threadgroup.width = 1;
	threads_threadgroup.depth = 1;

	threadgroup_per_grid.height = 5;
	threadgroup_per_grid.width  = 5;
	threadgroup_per_grid.depth = 1;

	// perform computation	
	// dispatch threads to GPU
	encoder->dispatchThreadgroups(threadgroup_per_grid, threads_threadgroup);
	encoder->endEncoding();
	// commit jobs 
	commandBuffer->commit();
	// let CPU wait on GPU tasks
	commandBuffer->waitUntilCompleted();

	// print output contents (viz)
	float* output_buffer = (float*)(buff_out->contents());
	
	// print out buffer values
	for(int i = 0; i < total_el_size; i++) {
		std::cout << output_buffer[i] << "\n";
	}
*/
	dev->release();


}



