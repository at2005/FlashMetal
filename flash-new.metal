
kernel void attention(const device float* query[[buffer(0)]], const device float* key[[buffer(1)]], const device float* value[[buffer(2)]], 
device float* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]]) 


{ 

	// we will deal with a single query, and multiple keys through a loop. But first we gotta copy part of the keys to SRAM

	const unsigned int query_size = 4;
	const unsigned int key_size = 4;
	const unsigned int embed_dim = 128;
	const unsigned int seq_len = 64;
	const unsigned int num_keys = seq_len / key_size;
	
	//  stored in SRAM
	threadgroup float KEY_SRAM[seq_len * embed_dim]; 
	threadgroup float VALUE_SRAM[seq_len * embed_dim];
	threadgroup float QUERY_SRAM[seq_len * embed_dim];

	// copy all keys and values to SRAM
	unsigned int elements_to_copy = query_size * embed_dim;
	for(int k = 0; k < elements_to_copy; k++) {	
		QUERY_SRAM[tid.y * elements_to_copy + k] = query[tid.y * elements_copy + k];		
		KEY_SRAM[tid.y * elements_to_copy + k] = key[tid.y * elements_copy + k];		
		VALUE_SRAM[tid.y * elements_to_copy + k] = value[tid.y * elements_copy + k];		
		
	}
		
	// ensure all threads finish copying before usage
	threadgroup_barrier(metal::mem_flags::mem_threadgroup);

	// iterate over each key and compute attention scores
	for(int k = 0; k < num_keys; k++) {

		// do matmul -- outer loop is each row in Q-block
		for(int i = 0; i < query_size; i++) {
			// inner loop is each row in K-block. Should be column but it's transposed
			for(int j = 0; j < key_size; j++) {
				// compute dot product
				float total_dot = 0.0;
				for(int el = 0; el < embed_dim; el++) { 
					// the logic here is that we first index the query into the particular block, then 
					// into the particular row (by i), then get the particular element by adding the offset.
					// for the key, we first index by k to isolate the block, then by j to get the row, and then by el.
					total_dot += QUERY_SRAM[tid.y * embed_dim + (i*embed_dim) + el] * KEY_SRAM[k*embed_dim + (j*embed_dim) + el];
				}
				
				// each query vector adds another row to the output attention scores
				out[tid.y * seq_len + (i*seq_len) + j] = total_dot;

			}

		}

	}


}
