
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t MovAvgWithCuda(float *result, const float *input, size_t size);

__global__ void MovAvgKernel(float *result, const float *input)
{
    int i = threadIdx.x;
    result[i] = input[i];
}

int main()
{
    const int arraySize = 10000;

	const int avgWindowSize = 15;

	float a[arraySize] = {0};
	for(int i=0; i<arraySize; i++)
	{
		a[i] = i;
	}

    float result[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = MovAvgWithCuda(result, a, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	printf("AllGood");
	getchar();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t MovAvgWithCuda(float *result, const float *input, size_t size)
{
	const int BLOCKS = 1;
	const int THREADS = 256;

    float *dev_input = 0;
	int dev_avgWindowSize = 0;
    float *dev_result = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_result, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_input, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    MovAvgKernel<<<1, 256>>>(dev_result, dev_input);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, dev_result, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_result);
    cudaFree(dev_input);
    
    return cudaStatus;
}
