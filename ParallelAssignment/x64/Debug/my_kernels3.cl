// The min/max kernels find the min/max value in the input vector A, by using a reduction pattern 
// i.e: with min, compare two values from the vector and saving the smallest one.
//This repeats until the total smallest value is stored at the first index position of the returned vector B. 

__kernel void reduce_find_min(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] = (scratch[lid] < scratch[lid + i]) ? scratch[lid] : scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid)
	{
		atomic_min(&B[0], scratch[lid]);
	}
}

///
///	This kernel works exactly the same as the min value finder, except with the < symbol reversed
///
__kernel void reduce_find_max(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] = (scratch[lid] > scratch[lid + i]) ? scratch[lid] : scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid)
	{
		atomic_max(&B[0], scratch[lid]);
	}
}

__kernel void reduce_find_sum(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid)
	{
		atomic_add(&B[0], scratch[lid]);
	}
}

__kernel void reduce_find_sum_var(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// This is because the values are x 100 when read in from file to preserve data
	// Therefore 100 ^ 2 = 10,000 which needs to be shrunk back to normal size again
	scratch[lid] = scratch[lid] / 10000.0f;

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid)
	{
		atomic_add(&B[0], scratch[lid]);
	}
}

__kernel void find_var(__global const int* A, __global int* B, int mean, int initSize)
{
	int id = get_global_id(0);

	if (id < initSize)
	{
		B[id] = A[id] - mean;

		barrier(CLK_LOCAL_MEM_FENCE);

		B[id] = (B[id] * B[id]);
	}
}

__kernel void atom_find_min(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local memory

	atomic_min(&B[0], scratch[lid]);
}

__kernel void atom_find_max(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local memory

	atomic_max(&B[0], scratch[lid]);
}

// Compares and sets value for output B
//__kernel void pass(__global const int* A, __global int* B, __local int* scratch, int initSize)
//{
//	int id = get_global_id(0);
//	int lid = get_local_id(0);
//
//	scratch[lid] = A[id];
//
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	for (id < initSize)
//	{
//
//		if (A[id] > B[id])
//		{
//			B[id] = A[id];
//		}
//	}
//}
//
//__kernel void merge(__global const int* A, __global int* B, __local int* scratch, int initSize)