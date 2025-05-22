#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1024

//from NVIDIA, for error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void AtomicSum(const float *array, const int size, float *sum){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < size){
        atomicAdd(sum, array[idx]);
    }
}

int main(){
    
    float *h_arr, *d_arr;
    float h_sum_check = 0.0f;
    float *d_sum, h_sum;

    h_arr = new float[ARRAY_SIZE];
    for (int i=0; i<ARRAY_SIZE; i++){
        h_arr[i] = i; // sum is n(n-1)/2
        h_sum_check += h_arr[i];
    }

    cudaMalloc(&d_arr, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_arr, h_arr, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("allocated and pushed to device");

    const int threads_per_block = 256;
    const int blocks_per_grid = (ARRAY_SIZE + threads_per_block - 1) / threads_per_block;

    AtomicSum<<<blocks_per_grid, threads_per_block>>>(d_arr, ARRAY_SIZE, d_sum);
    cudaDeviceSynchronize();
    cudaCheckErrors("finished kernel");

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("result back on host");
    printf("Atomic sum on GPU: %f, sum on CPU: %f \n", h_sum, h_sum_check);

    free(h_arr); cudaFree(d_arr); cudaFree(d_sum);
    return 0;
}

