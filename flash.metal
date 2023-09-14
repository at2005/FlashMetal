
kernel void mat_mul(const device float* mat1 [[buffer(0)]], const device float* mat2 [[buffer(1)]], device float* out [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
        float total = 0;
        int i = gid.y;
        int j = gid.x;
        int ROW_SIZE = 5;
        for(int k = 0; k < ROW_SIZE; k++) {
                total += (mat1[ROW_SIZE * k + i] * mat2[j* ROW_SIZE + k]);
        }

        out[j*ROW_SIZE + i] = total; //metal::exp(total / 400);

}



kernel void attention(const device float* query[[buffer(0)]], const device float* key[[buffer(1)]], const device float* value[[buffer(2)]], const device float* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
        

        int i = gid.y;
        int j = gid.x;
        int ROW_SIZE = 5;
	
	int BLOCK_SIZE = 1;	
	
	unsigned int elements_to_copy = (ROW_SIZE / BLOCK_SIZE);
	
	// SRAM holds both key and query 
	const unsigned int sram_len = 2 * ROW_SIZE * BLOCK_SIZE;
	threadgroup float SRAM[400];
	unsigned int offset_block = ROW_SIZE * BLOCK_SIZE;	
	
	
	// initial partial transfer to SRAM
	for(int k = 0; k < elements_to_copy; k++) {
		unsigned int temp_ind = (tid.x * elements_to_copy) + k;
		SRAM[temp_ind] = query[temp_ind];
		SRAM[offset_block + temp_ind] = key[temp_ind];
	}

	threadgroup_barrier(mem_flags::mem_threadgroup);

	// trying to do matmul now...
	// so, basically I have to calculate a single dot product, with a single loop, for a pair of rows
	// I have to find a particular row in SRAM: sram[i * row_size + j], where i ranges from 0 to block size, to isolate row in block. Then j just 
	// gets us where we need to go locally within that row, so from 0-row_size
	// And for key I just need to add offset_block
	
	float total = 0;
	
	for(int j = 0; j < ROW_SIZE; j++) {
		total += SRAM[tid.x * BLOCK_SIZE + j] + SRAM[offset_block + (tid.y * BLOCK_SIZE + j)];	

	}

	/*
        for(int k = 0; k < ROW_SIZE; k++) {
                total += (query[ROW_SIZE * k + i] * key[j* ROW_SIZE + k]);
        }
		
        out[j*ROW_SIZE + i] = metal::exp(total); 
*/
}
