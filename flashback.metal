

#include <metal_stdlib>


void memcpy_hbm_to_local(thread float* output, const device float* input, unsigned int num_to_copy) {
	for(unsigned int i = 0; i < num_to_copy; i++) {
		output[i] = input[i];
	}

}

void memcpy_hbm_to_sram(threadgroup float* output, const device float* input, unsigned int num_to_copy) {
	for(unsigned int i = 0; i < num_to_copy; i++) {
		output[i] = input[i];
	}

}

void memcpy_sram_to_hbm(device float* output, const threadgroup float* input, unsigned int num_to_copy) {
	for(unsigned int i = 0; i < num_to_copy; i++) {
		output[i] = input[i];
	}

}


kernel void backprop_attention(const device float* query[[buffer(0)]], const device float* key[[buffer(1)]], const device float* value[[buffer(2)]], 
device float* out [[buffer(3)]], device float* dO [[buffer(4)]], device float* out_dV [[buffer(5)]], device float* ROW_SUMS [[buffer(6)]], device float* ROW_MAX_VALS [[buffer(7)]], uint2 gid [[thread_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]]) {
	
	
	const unsigned int query_size = 8;
	const unsigned int key_size = 8;
	const unsigned int embed_dim = 96;
	const unsigned int seq_len = 1024;
	const unsigned int num_keys = seq_len / key_size;
	const unsigned int num_heads = 1;

	const unsigned int num_values_batch = num_heads * seq_len * embed_dim;
	const unsigned int num_values_head = seq_len * embed_dim;


	const unsigned int batch_index = tgid.y * num_values_batch;
	const unsigned int head_index = tgid.x * num_values_head;

	const unsigned int num_el_kv = key_size * embed_dim;
	const unsigned int num_el_query = query_size * embed_dim;

	const unsigned int dV_elements = key_size * embed_dim;
	const unsigned int num_threads = seq_len / query_size;

	const unsigned int row_val_offset = (tgid.y * num_heads * seq_len) + (tgid.x * seq_len) + (key_size*tid.y);
	// Each thread adds num_el sets of elements, where num_el is number of elements in dV_i = embed_dim * key_size, divided by the total number of threads = (seq_len / query_size)
	// hence num_el = query_size * key_size * embed_dim / seq_len
	const unsigned int num_el = dV_elements / num_threads; 

	// IMPORTANT
	// tid.y contains the query index. Each threadgroup contains B_q query blocks, which compute a particular attention score. Each threadgroup contributes for one head in a single batch
	// tgid.x contains the current head dimension
	// tgid.y contains the current batch dimension
	
	// initialise buffers for copying -- not in SRAM as we do not wish to share it
	float QUERY_LOCAL[num_el_query];
	float OUTPUT_LOCAL[key_size * query_size];
	float dO_LOCAL[num_el_query];

	float dim_factor = metal::sqrt((float)embed_dim);

	// copy all queries/outputs to SRAM
	unsigned int elements_to_copy = query_size * embed_dim;
	
	for(int i = 0; i < elements_to_copy; i++) QUERY_LOCAL[i] = query[tid.y * elements_to_copy + i];

	memcpy_hbm_to_local(dO_LOCAL, dO + sizeof(float)*(batch_index + head_index + tid.y*elements_to_copy), elements_to_copy);
	
	threadgroup_barrier(metal::mem_flags::mem_threadgroup);

	unsigned int elements_key_copy = (key_size * key_size * embed_dim) / seq_len;
	
	const unsigned int local_offset_kv = tid.y * elements_key_copy;
	const unsigned int total_offset_bhkv = batch_index + head_index + local_offset_kv;
	
	
	threadgroup float KEY_SRAM[key_size * embed_dim];
	threadgroup float VALUE_SRAM[key_size * embed_dim];

	// SRAM contains sum of all dVs computed by each thread in group
	threadgroup float dV_acc[dV_elements];
	float dV[dV_elements];
	
	// iterate over each key block and compute attention scores
	for(unsigned int k = 0; k < num_keys; k++) {
		// copy from HBM, each thread copies a little bit of the shared key/value block into SRAM.	
		for(int i = 0; i < elements_key_copy; i++) {
			KEY_SRAM[local_offset_kv + i] = key[k*num_el_kv + total_offset_bhkv + i];
			VALUE_SRAM[local_offset_kv + i] = value[k*num_el_kv + total_offset_bhkv + i];
		}
		
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
		
		// this contains the attention score matrix before softmax

		// do matmul -- outer loop is each row in Q-block
		for(unsigned int i = 0; i < query_size; i++) {
			// inner loop is each row in K-block. Should be column but it's transposed
			for(unsigned int j = 0; j < key_size; j++) {
				/*	
				// for LT matrix
				if((tid.y * query_size + i) < (k * key_size + j)) {
					OUTPUT_LOCAL[i*query_size + j] = 0.0;
					continue;
				}*/

				// compute dot product
				float total_dot = 0.0;
				for(unsigned int el = 0; el < embed_dim; el++) { 
					// the logic here is that we first index the query into the particular block, then 
					// into the particular row (by i), then get the particular element by adding the offset.
					// for the key, we first index by k to isolate the block, then by j to get the row, and then by el.
					//total_dot += QUERY_LOCAL[(tid.y * query_size * embed_dim) + (i*embed_dim) + el] * KEY_SRAM[(k*key_size*embed_dim) + (j*embed_dim) + el];
					
					// shape of row-based tensors = (b, h, seq_len)
					// first index into batch. then head, then into block (part of seq len)
					total_dot += QUERY_LOCAL[i*embed_dim+ el] * KEY_SRAM[(j*embed_dim) + el];
				}
				
				// each query vector adds another row to the output attention scores
				OUTPUT_LOCAL[i*query_size + j] = total_dot / dim_factor;				
				OUTPUT_LOCAL[i*query_size + j] = metal::exp(OUTPUT_LOCAL[i*query_size + j] - ROW_MAX_VALS[row_val_offset + i]) / ROW_SUMS[row_val_offset + i];

				out[(tid.y * query_size + i) * seq_len + k*key_size + j] = OUTPUT_LOCAL[i*query_size + j];

			}

		}
		
			
		// compute dV_part = P^T dO
		// == each column of OUTPUT_LOCAL dotted with each row of dO_LOCAL

		// iterate over each column (OUTPUT_LOCAL is of shape (query_size, key_size)), and dO is of shape (query_size, embed_dim)
		for(unsigned int o_col = 0; o_col < query_size; o_col++) {
			for(unsigned int dO_col = 0; dO_col < embed_dim; dO_col++) {
				// dot product
				float total_dot = 0.0;
				for(unsigned int el = 0; el < query_size; el++) {
					total_dot += OUTPUT_LOCAL[el*key_size + o_col] * dO_LOCAL[el*embed_dim + dO_col];
				}
				
				dV[o_col*embed_dim + dO_col] = total_dot;	

			}
		}
	
		// all threads must finish computation
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
		
		// time to add!
		// want to accumulate in a multi-threaded way
		// basically chunk space up and iterate over each chunk, each thread writes particular bit into chunk
		
		// iteration zero -> thread zero writes to zeroth block here (each block is num_el elements)
		// thread one writes to first etc etc
		// iteration one -> thread zero write to first block, first block of its own matrix to first block of acc matrix
		
		// we get a pattern where tid.y is used to index into the appropriate block for each iteration
		// for zeroth iteration, we have tid.y*num_el to index into, and then we just add the current iteration counter so ((tid.y+i) % num_threads) * num_el
		// equivalent to reshaping matrix as (num_el, num_threads)
		
		for(int i = 0; i < num_el; i++) dV_acc[tid.y * num_el + i] = 0.0;
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);

		for(int i = 0; i < num_threads; i++) {
			unsigned int rotated_block_index = ((tid.y + i) % num_threads) * num_el;
			for(int el_acc = 0; el_acc < num_el; el_acc++) {
				dV_acc[rotated_block_index + el_acc] += dV[rotated_block_index + el_acc];	
				threadgroup_barrier(metal::mem_flags::mem_threadgroup);

			}
		}
		
		// now we want to copy this to an output tensor
//		memcpy_sram_to_hbm(out_dV + sizeof(float)*(batch_index + head_index + (tid.y*dV_elements)), dV, dV_elements); 
		
		for(int i = 0; i < num_el; i++) out_dV[k*dV_elements + tid.y * num_el + i] = dV_acc[tid.y * num_el + i];

		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
	
		
		

	}
	

}
