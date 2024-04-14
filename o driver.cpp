

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

	// Currently assume the existence of ROW_MAX, an array of dim (num query rows,1 )


	// IMPORTANT
	// tid.y contains the query index. Each threadgroup contains B_q query blocks, which compute a particular attention score. Each threadgroup contributes for one head in a single batch
	// tgid.x contains the current head dimension
	// tgid.y contains the current batch dimension
	
	// initialise buffers for copying -- not in SRAM as we do not wish to share it
	float QUERY_LOCAL[num_el_query];
	float OUTPUT_LOCAL[num_el_query];
	float dO_LOCAL[num_el_query];

	float dim_factor = metal::sqrt((float)embed_dim);

	// copy all queries/outputs to SRAM
	unsigned int elements_to_copy = query_size * embed_dim;
		
	memcpy_hbm_to_local(QUERY_LOCAL, query + sizeof(float)*(batch_index + head_index + tid.y*elements_to_copy), elements_to_copy);
	memcpy_hbm_to_local(OUTPUT_LOCAL, out + sizeof(float)*(batch_index + head_index + tid.y*elements_to_copy), elements_to_copy);
	memcpy_hbm_to_local(dO_LOCAL, dO + sizeof(float)*(batch_index + head_index + tid.y*elements_to_copy), elements_to_copy);
		
	unsigned int elements_key_copy = (key_size * key_size * embed_dim) / seq_len;
	
	const unsigned int local_offset_kv = tid.y * elements_key_copy;
	const unsigned int total_offset_bhkv = batch_index + head_index + local_offset_kv;
	
	
	threadgroup float KEY_SRAM[key_size * embed_dim];
	threadgroup float VALUE_SRAM[key_size * embed_dim];

	// SRAM contains all dVs computed by each thread in group
	 threadgroup float dV[num_threads * dV_elements];

	// iterate over each key block and compute attention scores
	for(unsigned int k = 0; k < num_keys; k++) {
		// copy from HBM, each thread copies a little bit of the shared key/value block into SRAM.
		memcpy_hbm_to_sram(KEY_SRAM + sizeof(float)*local_offset_kv, key + sizeof(float)*(total_offset_bhkv + (k*num_el_kv)),  elements_key_copy);
		memcpy_hbm_to_sram(VALUE_SRAM + sizeof(float)*local_offset_kv, value + sizeof(float)*(total_offset_bhkv + (k*num_el_kv)),  elements_key_copy);

		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
		
		// this contains the attention score matrix before softmax
		float OUTPUT_LOCAL[query_size * key_size];

		// do matmul -- outer loop is each row in Q-block
		for(unsigned int i = 0; i < query_size; i++) {
			// inner loop is each row in K-block. Should be column but it's transposed
			for(unsigned int j = 0; j < key_size; j++) {
				
				// for LT matrix
				if((tid.y * query_size + i) < (k * key_size + j)) {
					OUTPUT_LOCAL[i*query_size + j] = 0.0;
					continue;
				}

				// compute dot product
				float total_dot = 0.0;
				for(unsigned int el = 0; el < embed_dim; el++) { 
					// the logic here is that we first index the query into the particular block, then 
					// into the particular row (by i), then get the particular element by adding the offset.
					// for the key, we first index by k to isolate the block, then by j to get the row, and then by el.
					//total_dot += QUERY_LOCAL[(tid.y * query_size * embed_dim) + (i*embed_dim) + el] * KEY_SRAM[(k*key_size*embed_dim) + (j*embed_dim) + el];
					
					// shape of row-based tensors = (b, h, seq_len)
					// first index into batch. then head, then into block (part of seq len)
					unsigned int row_val_offset = (tgid.y * num_heads * seq_len) + (tgid.x * seq_len) + (key_size*tid.y);
					total_dot += metal::exp((QUERY_LOCAL[i*embed_dim+ el] * KEY_SRAM[(j*embed_dim) + el]) - ROW_MAX_VALS[row_val_offset + i]) / ROW_SUMS[row_val_offset + i];

				}
				
				// each query vector adds another row to the output attention scores
				OUTPUT_LOCAL[i*query_size + j] = total_dot / dim_factor;				

			}

		}
		
		
		// compute dV_part = P^T dO
		// == each column of OUTPUT_LOCAL dotted with each row of dO_LOCAL
			
		 const unsigned int dV_elements = key_size * embed_dim;
		 const unsigned int num_threads = seq_len / query_size;


		// iterate over each column (OUTPUT_LOCAL is of shape (query_size, key_size)), and dO is of shape (query_size, embed_dim)
		for(unsigned int o_col = 0; o_col < query_size; o_col++) {
			for(unsigned int dO_col = 0; dO_col < embed_dim; dO_col++) {
				// dot product
				float total_dot = 0.0;
				for(unsigned int el = 0; el < query_size; el++) {
					total_dot += OUTPUT_LOCAL[el*key_size + o_col] * dO_LOCAL[el*embed_dim + dO_col];
				}
				
				dV[tid.y * dV_elements + o_col*embed_dim + dO_col] = total_dot;	

			}
		}
	
		// all threads must finish computation
		threadgroup_barrier(metal::mem_flags::mem_threadgroup);

		// perform parallel cumsum
		// Each thread adds num_el sets of elements, where num_el is number of elements in dV_i = embed_dim * query_size, divided by the total number of threads = (seq_len / query_size)
		// hence num_el = query_size^2 * embed_dim / seq_len
		const unsigned int num_el = (query_size * query_size * embed_dim) / seq_len;

		// time to add!
		// each thread to index into dV and add all the other corresponding elements
		// the logic here is that we only iterate over each set of num_el elements in the first dV block
		// Then, for each of those, we accumulate all other dVs (1, 2, .. num_threads - 1) into zeroth block 

		for(unsigned int idx_in_set = 0; idx_in_set < num_el; idx_in_set++) {
			// starts at zero technically but we need to offset by the current thread division
			unsigned int offset_index = idx_in_set + tid.y * num_el;
			// gotta accumulate 
			for(unsigned int dV_index = 0; dV_index < num_threads; dV_index++) {
	//			dV[offset_index] += dV[dV_index * dV_elements + idx_in_set];

			}

		}

		
		// now we want to copy this to an output tensor
//		memcpy_sram_to_hbm(out_dV + sizeof(float)*(batch_index + head_index + (tid.y*dV_elements)), dV, dV_elements); 
		

		for(int out_i = 0; out_i < 96*1024; out_i++) {
			//out_dV[(k * dV_elements) + (tid.y * num_el) + out_i] = dV[out_i] + 10; 
			
		}
	
		
		

	}


}
