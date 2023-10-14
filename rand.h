
class CustomRandom {
	public:	
		CustomRandom(long seed);
		float generate();
	
	private:
		long state;
		long seed;
		long a;
		long c;
		long m;
};
