
kernel void mat_mul(const device float* mat1 [[buffer(0)]], const device float* mat2 [[buffer(1)]], const device float* out [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
	float total = 0;
	int i = gid.x;
	int j = gid.y;
	int ROW_SIZE = 5;
	for(int k = 0; k < 5; k++) {
		total += A[i + k * ROW_SIZE] * B[j*ROW_SIZE + k];
	}

	C[i*ROW_SIZE + j] = total;

}
