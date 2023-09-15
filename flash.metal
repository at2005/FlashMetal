
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



kernel void attention(const device float* query[[buffer(0)]], const device float* key[[buffer(1)]], const device float* value[[buffer(2)]], device float* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]]) {
  


        int ROW_SIZE = 6;
	
	int BLOCK_SIZE = 2;	
	
	unsigned int elements_to_copy = (ROW_SIZE / BLOCK_SIZE);
	
	// SRAM holds both key and query 
	const unsigned int sram_len = 2 * ROW_SIZE * BLOCK_SIZE;
	threadgroup float SRAM[100];
	unsigned int offset_block = ROW_SIZE * BLOCK_SIZE;	

	unsigned int abs_tid = tid.x * BLOCK_SIZE + tid.y;

	unsigned int offset_gblock_x = tgid.x * offset_block;
	unsigned int offset_gblock_y = tgid.y * offset_block;
	
	// initial partial transfer to SRAM
	for(int k = 0; k < elements_to_copy; k++) {
		unsigned int temp_ind = (abs_tid * elements_to_copy) + k;
		SRAM[temp_ind] = query[offset_gblock_x + temp_ind];
		SRAM[offset_block + temp_ind] = key[offset_gblock_y + temp_ind];
	}

	threadgroup_barrier(metal::mem_flags::mem_threadgroup);

	// trying to do matmul now...
	// so, basically I have to calculate a single dot product, with a single loop, for a pair of rows
	// I have to find a particular row in SRAM: sram[i * row_size + j], where i ranges from 0 to block size, to isolate row in block. Then j just 
	// gets us where we need to go locally within that row, so from 0-row_size
	// And for key I just need to add offset_block
	// ... and also add a global offset, ie threadgroup position offset so that correct and unique block copied 

	float total = 0;
	
	for(int j = 0; j < ROW_SIZE; j++) {
		total += SRAM[tid.x * ROW_SIZE + j] * SRAM[offset_block + (tid.y * ROW_SIZE + j)];	
	}

        out[gid.x*ROW_SIZE + gid.y] = total; 
	

}
