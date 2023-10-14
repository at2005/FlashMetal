
kernel void attention(const device float* query[[buffer(0)]], const device float* key[[buffer(1)]], const device float* value[[buffer(2)]], 
device float* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]]) 


{ 

	// we will deal with a single query, and multiple keys through a loop. But first we gotta copy part of the keys to SRAM

	const unsigned int query_size = 4;
	const unsigned int key_size = 4;
	const unsigned int embed_dim = 464;
	const unsigned int seq_len = 464;
	const unsigned int num_keys = seq_len / key_size;


	//  stored in SRAM
	//threadgroup float KEY_SRAM[seq_len * embed_dim]; 
	//threadgroup float VALUE_SRAM[seq_len * embed_dim];

	//threadgroup float QUERY_SRAM[seq_len * embed_dim];
	float QUERY_SRAM[query_size * embed_dim];

	float dim_factor = metal::sqrt((float)embed_dim);

	// copy all keys and values to SRAM
	unsigned int elements_to_copy = query_size * embed_dim;
	for(int k = 0; k < elements_to_copy; k++) {	
		QUERY_SRAM[k] = query[tid.y * elements_to_copy + k];		

		//QUERY_SRAM[tid.y * elements_to_copy + k] = query[tid.y * elements_to_copy + k];		
//		KEY_SRAM[tid.y * elements_to_copy + k] = key[tid.y * elements_to_copy + k];		
//		VALUE_SRAM[tid.y * elements_to_copy + k] = value[tid.y * elements_to_copy + k];		
		
	}
		
	// ensure all threads finish copying before usage

	float ROW_SUM[query_size];
	float ROW_MAX[query_size];
	for(int _ = 0; _ < query_size; _++) {
		ROW_SUM[_] = 0.0; 
		ROW_MAX[_] = -INFINITY;
	}
	
	// size = seq_len / num_threads
	// each query must copy 1/num_threads of key/value which are size*embed_dim
	// so because each key/val tensor is

	unsigned int elements_key_copy = (key_size * key_size * embed_dim) / seq_len;
	
	// iterate over each key block and compute attention scores
	for(int k = 0; k < num_keys; k++) {
		threadgroup float KEY_SRAM[key_size * embed_dim];
		threadgroup float VALUE_SRAM[key_size * embed_dim];
		// copy values from HBM, each thread copies a little bit of the shared key/value tensor
		for(int _ = 0; _ < elements_key_copy; _++) {
			KEY_SRAM[tid.y * elements_key_copy + _] = key[(k*key_size*embed_dim) + (tid.y * elements_key_copy) +  _]; 
			VALUE_SRAM[tid.y * elements_key_copy + _] = value[(k*key_size*embed_dim) + (tid.y * elements_key_copy) +  _]; 
		}

		threadgroup_barrier(metal::mem_flags::mem_threadgroup);
		
		float OUTPUT_SRAM[query_size * key_size];
		float ROW_MAX_LOCAL[query_size];
		for(int _ = 0; _ < query_size; _++) ROW_MAX_LOCAL[_] = -INFINITY;

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
					//total_dot += QUERY_SRAM[(tid.y * query_size * embed_dim) + (i*embed_dim) + el] * KEY_SRAM[(k*key_size*embed_dim) + (j*embed_dim) + el];
					total_dot += QUERY_SRAM[i*embed_dim+ el] * KEY_SRAM[(j*embed_dim) + el]; 

				}
				
				// each query vector adds another row to the output attention scores
				total_dot /= dim_factor;
				if(total_dot > ROW_MAX_LOCAL[i]) ROW_MAX_LOCAL[i] = total_dot;
				OUTPUT_SRAM[i*query_size + j] = total_dot;				

			}

		}

		
		float ROW_SUM_NEW[query_size];
		// calculating row_sums and exponentiating with maximum
		for(int row_i = 0; row_i < query_size; row_i++) {
			float row_sum_local = 0.0;	

			for(int row_el = 0; row_el < key_size; row_el++) {
				size_t output_index = row_i * query_size + row_el;
				if(tid.y * query_size + (row_i) < (k * key_size+ row_el)) {
					OUTPUT_SRAM[output_index] = 0.0;
				//	test[(tid.y * seq_len*query_size) + (row_i*seq_len) + (k * key_size+ row_el)] = 0.0; 
					continue;
				}
				
				//test[(tid.y * seq_len*query_size) + (row_i*seq_len) + (k * key_size+ row_el)] = OUTPUT_SRAM[output_index];

				// compute exponent with stability
//				if(OUTPUT_SRAM[output_index] == -INFINITY) OUTPUT_SRAM[output_index] = 0.0;
				OUTPUT_SRAM[output_index] = metal::exp(OUTPUT_SRAM[output_index] - ROW_MAX_LOCAL[row_i]);
				row_sum_local += OUTPUT_SRAM[output_index];	
			}
			
			ROW_SUM_NEW[row_i] = row_sum_local;

		}
		
		
		// computing value dot attention scores
		// so, basically, iterate over each row in attention scores, for us that is reduced seq dimension
		for(int att_row = 0; att_row < query_size; att_row++) {
			
			float rowmax_new = metal::max(ROW_MAX[att_row], ROW_MAX_LOCAL[att_row]);
			float sum_divisor = ((metal::exp(ROW_MAX[att_row] - rowmax_new) * ROW_SUM[att_row]) + (metal::exp(ROW_MAX_LOCAL[att_row] - rowmax_new) * ROW_SUM_NEW[att_row]));


			// iterate over each column in value matrix
			for(int val_col = 0; val_col < embed_dim; val_col++) {

				size_t outmat_i = (embed_dim * tid.y * query_size) + (att_row*embed_dim) + val_col;

				float val_dot = 0.0;
				// dot prod computation
				for(int el = 0; el < query_size; el++) {
					val_dot += OUTPUT_SRAM[(att_row * query_size) + el] * VALUE_SRAM[val_col + (el*embed_dim)];//* VALUE_SRAM[(k*key_size*embed_dim) + val_col + (el*embed_dim)];
				}
				
//				out[outmat_i] = 1.4323;

				// multiply to cancel out the previous incorrect row sums
				out[outmat_i] *= ROW_SUM[att_row];
				// multiply by e^old_max to cancel and -e to include new max	
				out[outmat_i] *= metal::exp(ROW_MAX[att_row] - rowmax_new);
				// add new score value to SV dot product
				out[outmat_i] += metal::exp(ROW_MAX_LOCAL[att_row] - rowmax_new) * val_dot;
				out[outmat_i] /= sum_divisor;

			}
			
			// update rowsum
			ROW_SUM[att_row] = sum_divisor;//(metal::exp(ROW_MAX[att_row] - rowmax_new) * ROW_SUM[att_row] + metal::exp(ROW_MAX_LOCAL[att_row] - rowmax_new) * ROW_SUM_NEW[att_row]);
			ROW_MAX[att_row] = rowmax_new;
		


		}


	}


}
