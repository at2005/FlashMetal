
kernel void attention(const device float* query[[buffer(0)]], const device float* key[[buffer(1)]], const device float* value[[buffer(2)]], 
device float* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]]) 


{ 

	// we will deal with a single query, and multiple keys through a loop. But first we gotta copy part of the keys to SRAM

	const unsigned int query_size = 4;
	const unsigned int key_size = 4;
	const unsigned int embed_dim = 32;
	const unsigned int seq_len = 64;
	const unsigned int num_keys = seq_len / key_size;


	//  stored in SRAM
	threadgroup float KEY_SRAM[seq_len * embed_dim]; 
	threadgroup float VALUE_SRAM[seq_len * embed_dim];
	threadgroup float QUERY_SRAM[seq_len * embed_dim];
	
	float dim_factor = 1.0;//metal::sqrt(embed_dim);


	// copy all keys and values to SRAM
	unsigned int elements_to_copy = query_size * embed_dim;
	for(int k = 0; k < elements_to_copy; k++) {	
		QUERY_SRAM[tid.y * elements_to_copy + k] = query[tid.y * elements_to_copy + k];		
		KEY_SRAM[tid.y * elements_to_copy + k] = key[tid.y * elements_to_copy + k];		
		VALUE_SRAM[tid.y * elements_to_copy + k] = value[tid.y * elements_to_copy + k];		
		
	}
		
	// ensure all threads finish copying before usage
	threadgroup_barrier(metal::mem_flags::mem_threadgroup);

	// iterate over each key block and compute attention scores
	for(int k = 0; k < num_keys; k++) {
		float OUTPUT_SRAM[query_size * key_size];
		// do matmul -- outer loop is each row in Q-block
		for(int i = 0; i < query_size; i++) {
			// inner loop is each row in K-block. Should be column but it's transposed
			for(int j = 0; j < key_size; j++) {
				
				// output matrix has materialised
				
				// compute dot product
				float total_dot = 0.0;
				for(int el = 0; el < embed_dim; el++) { 
					// the logic here is that we first index the query into the particular block, then 
					// into the particular row (by i), then get the particular element by adding the offset.
					// for the key, we first index by k to isolate the block, then by j to get the row, and then by el.
					total_dot += QUERY_SRAM[(tid.y * query_size * embed_dim) + (i*embed_dim) + el] * KEY_SRAM[(k*key_size*embed_dim) + (j*embed_dim) + el];
				}
				
				// each query vector adds another row to the output attention scores
				
				OUTPUT_SRAM[i*query_size + j] = total_dot / dim_factor;				
					

//				out[(tid.y * query_size * seq_len) + (i*seq_len) + (k*key_size) + j] = OUTPUT_SRAM[i*query_size + j];//metal::exp(total_dot / 200);

				//out[tid.y] = query[embed_dim*seq_len - 1];//QUERY_SRAM[embed_dim * seq_len - 1];

			}

		}

		
		// computing value dot attention scores

		// so, basically, iterate over each row in attention scores, for us that is reduced seq dimension
		for(int att_row = 0; att_row < query_size; att_row++) {
			// iterate over each column in value matrix
			for(int val_col = 0; val_col < embed_dim; val_col++) {
					
				float val_dot = 0.0;
				// dot prod computation
				for(int el = 0; el < query_size; el++) {
					val_dot += OUTPUT_SRAM[(att_row * query_size) + el] * VALUE_SRAM[(k*key_size*embed_dim) + val_col + (el*embed_dim)];
				}
				
				out[(embed_dim * tid.y * query_size) + (att_row*embed_dim) + val_col] += val_dot;


			}
		}


	}


}
