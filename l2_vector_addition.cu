#include <stdio.h>
#include <iostream>
#include <stdlib.h> // for rand()

const int num_elements = 10000;

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

    //CUDA kernel to add two vectors
__global__ void vector_addition(const float *A, const float *B, float *C, int array_size){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (array_size>blockDim.x*gridDim.x && idx==0){ //check if we have enough threads and print error once
        printf("Error: array size is larger than the total number of threads\n");
        return;
    }
    if (idx < array_size){
        C[idx] = A[idx] + B[idx];
    }
}



int main() {


    // device array of num_elements elements
    float *d_a, *d_b, *d_c, *h_a, *h_b, *h_c, *h_check; //device and host array pointers

    //load some values to host arrays
    h_a= (float*)malloc(num_elements*sizeof(float));
    h_b= (float*)malloc(num_elements*sizeof(float));
    h_c= (float*)malloc(num_elements*sizeof(float));
    h_check= (float*)malloc(num_elements*sizeof(float));
    for (int i=0; i<num_elements; i++){
        h_a[i] = rand();
        h_b[i] = rand();
        h_check[i] = h_a[i] + h_b[i]; // to verify the reults
    }
    
    //allocate device i.e. GPU arrays
    cudaMalloc(&d_a, num_elements*sizeof(float));
    cudaMalloc(&d_b, num_elements*sizeof(float));
    cudaMalloc(&d_c, num_elements*sizeof(float));
    cudaCheckErrors("cudaMalloc of device arrays a,b and c");

    //push host arrays a and b to device
    cudaMemcpy(d_a, h_a, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("pushed host arrays a and b to device");

    // kernel launch
    const int threads_per_block = 1024;
    //const int total_blocks = ceil(num_elements/threads_per_block); quick lesson: dont do this,int/int is round down already
    const int total_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    dim3 grid_dim(total_blocks,1,1); //onjy useing x grid and block for now
    dim3 block_dim(threads_per_block,1,1);
    vector_addition<<<grid_dim, block_dim>>>(d_a,d_b,d_c, num_elements);
    //vector_addition<<<1024, 16>>>(d_a,d_b,d_c, num_elements);
    cudaCheckErrors("finished vector addition kernel");

    //get back the c array on device
    cudaMemcpy(h_c, d_c, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("get back array c result to host");

    for (int i=0; i<num_elements; i++){
        if (h_c[i]!=h_check[i]){
            printf("Check index %d",i);
            throw std::runtime_error("Error: The result does not match");
        }
    }


    return 0;
}