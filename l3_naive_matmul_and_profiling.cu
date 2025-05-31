// porfiling flag: -lnvToolsExt
//nsys profile --stats=true ./out > l3_naive_matmul_profile_output.txt  -> profile entire application
//ncu -o profile ./out -> for specifically profiling CUDA kernels
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

#define PROFILING

#ifdef PROFILING
    #include <nvtx3/nvToolsExt.h>
#endif


const int matrix_size = 1000; //square matrix


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

__global__ void naive_matmul(const double *A, const double *B, double *C, const int arr_size){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if ((row<arr_size) && (col<arr_size)){
        double _temp=0.0f;
        for (int k=0; k<arr_size; k++){
            _temp += A[row*arr_size+k]*B[k*arr_size+col];
        }
        C[row*arr_size+col]= _temp;
    }

}

int main(){

    cudaEvent_t start, stop; //for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *h_check;
    double temp;
    h_A = new double[matrix_size*matrix_size];
    h_B = new double[matrix_size*matrix_size];
    h_C = new double[matrix_size*matrix_size];
    h_check = new double[matrix_size*matrix_size];
    for (int i = 0; i < matrix_size*matrix_size; i++){ //2D matrix are assumed row-major
        h_A[i] = (rand()/(float)RAND_MAX)*1000; // bugfix:if not casted to float, int/int will become 0
        h_B[i] = (rand()/(float)RAND_MAX)*1000; //without bounding, even double precision results wont match 
    }
    for (int i=0; i<matrix_size; i++){ // row index
        for (int j=0; j<matrix_size; j++){ // column inex
            temp = 0.0f;
            for (int k=0; k<matrix_size; k++){ // col of A, row of B
                // row*size+col; 
                temp += h_A[i*matrix_size+k] * h_B[k*matrix_size+j];
            }
            h_check[i*matrix_size+j] = temp; //get ans on cpu to verify gpu result
        }
    }
    #ifdef PROFILING
        nvtxRangePush("Array allocation");
    #endif
    cudaMalloc(&d_A, matrix_size*matrix_size*sizeof(double));
    cudaMalloc(&d_B, matrix_size*matrix_size*sizeof(double));
    cudaMalloc(&d_C, matrix_size*matrix_size*sizeof(double));
    #ifdef PROFILING
        nvtxRangePop();
    #endif

    #ifdef PROFILING
        nvtxRangePush("Push Matrix to device");
    #endif
    cudaMemcpy(d_A, h_A, matrix_size*matrix_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size*matrix_size*sizeof(double), cudaMemcpyHostToDevice);
    #ifdef PROFILING
        nvtxRangePop();
    #endif
    cudaCheckErrors("allocated and pushed to device");

    #ifdef PROFILING
    nvtxRangePush("Kernel launch");
    #endif
    const int thread_per_block = 32;
    const int block_per_grid = (matrix_size+thread_per_block-1)/(thread_per_block);
    dim3 block_dim(thread_per_block, thread_per_block,1);
    dim3 grid_dim(block_per_grid, block_per_grid, 1);
    cudaEventRecord(start);
    naive_matmul<<<grid_dim, block_dim>>>(d_A, d_B, d_C, matrix_size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    #ifdef PROFILING
        nvtxRangePop();
    #endif

    cudaCheckErrors("matmul kernel completed");

    #ifdef PROFILING
        nvtxRangePush("Copy result back to host");
    #endif
    cudaMemcpy(h_C, d_C, matrix_size*matrix_size*sizeof(double), cudaMemcpyDeviceToHost);
    #ifdef PROFILING
        nvtxRangePop();
    #endif
    cudaCheckErrors("result is back on host");

    float runtime = 0;
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Kernel execution time: %f ms\n", runtime);

    //verify the result
    for (int i=0; i<matrix_size*matrix_size; i++){ // row index
        //for (int j=0; j<matrix_size; j++){ // column inex
            //int index = i*matrix_size+j;
            if (i<5){printf("%f %f \n", h_C[i], h_check[i]);}; //for extra sanity check
           if(h_check[i] != h_C[i]){
            printf("result at index %d does not match, expected %f, got %f\n", i, h_check[i], h_C[i]);
            return 1;
           }
        }
    printf("Passed! \n");


    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_check;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}