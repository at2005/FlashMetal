
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



kernel void attention(const device float* query[[buffer(0)]], const device float* key[[buffer(1)]], const device float* value[[buffer(2)]], device float* out [[buffer(3)]], uint2 gid [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
        

	int BLOCK_SIZE = 1;	
	float total = 0;
	int MATRIX_SIZE = 5;

	threadgroup float SRAM[MATRIX_SIZE * MATRIX_SIZE]; 
	for(int i = tid; i < 	
	


        int i = gid.y;
        int j = gid.x;
        int ROW_SIZE = 5;
        for(int k = 0; k < ROW_SIZE; k++) {
                total += (query[ROW_SIZE * k + i] * key[j* ROW_SIZE + k]);
        }
		
        out[j*ROW_SIZE + i] = metal::exp(total); 

}
