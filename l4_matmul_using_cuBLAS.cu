// similar fo l3 but using cuBLAS for matrix multiplication
//cublas + profiling flags: -lcublas -lnvToolsExt
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include<cmath>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define PROFILING

#ifdef PROFILING
    #include <nvtx3/nvToolsExt.h>
#endif


const int matrix_size = 2000; //square matrix


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


int main(){

    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *h_check;
    float temp;
    h_A = new float[matrix_size*matrix_size];
    h_B = new float[matrix_size*matrix_size];
    h_C = new float[matrix_size*matrix_size];
    h_check = new float[matrix_size*matrix_size];
    for (int i = 0; i < matrix_size*matrix_size; i++){
        h_A[i] = (rand()/(float)RAND_MAX)*1; 
        h_B[i] = (rand()/(float)RAND_MAX)*1;
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
    cudaMalloc(&d_A, matrix_size*matrix_size*sizeof(float));
    cudaMalloc(&d_B, matrix_size*matrix_size*sizeof(float));
    cudaMalloc(&d_C, matrix_size*matrix_size*sizeof(float));
    #ifdef PROFILING
        nvtxRangePop();
    #endif

    #ifdef PROFILING
        nvtxRangePush("Push Matrix to device");
    #endif
    //cudaMemcpy(d_A, h_A, matrix_size*matrix_size*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, h_B, matrix_size*matrix_size*sizeof(float), cudaMemcpyHostToDevice);
    cublasSetVector(matrix_size*matrix_size, sizeof(float), h_A, 1, d_A, 1);
    cublasSetVector(matrix_size*matrix_size, sizeof(float), h_B, 1, d_B, 1);
    cublasSetVector(matrix_size*matrix_size, sizeof(float), h_C, 1, d_C, 1);
    #ifdef PROFILING
        nvtxRangePop();
    #endif
    cudaCheckErrors("allocated and pushed to device");

    #ifdef PROFILING
    nvtxRangePush("Kernel launch");
    #endif
   
    cudaEventRecord(start);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N, // N or T ; BLAS in fortran is column major
        matrix_size,
        matrix_size,
        matrix_size,
        &alpha,
        d_B,
        matrix_size,
        d_A,
        matrix_size,
        &beta,
        d_C,
        matrix_size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    #ifdef PROFILING
        nvtxRangePop();
    #endif

    cudaCheckErrors("matmul kernel completed");

    #ifdef PROFILING
        nvtxRangePush("Copy result back to host");
    #endif
    //cudaMemcpy(h_C, d_C, matrix_size*matrix_size*sizeof(float), cudaMemcpyDeviceToHost);
    cublasGetVector(matrix_size*matrix_size, sizeof(float), d_C, 1, h_C, 1);

    #ifdef PROFILING
        nvtxRangePop();
    #endif
    cudaCheckErrors("result is back on host");

    float runtime = 0;
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Kernel execution time: %f ms\n", runtime);
    
    //verify the result
    float tol = 1e-2f; ////cuBLAS single precision is accurate upto 7 digits i think 
    for (int i=0; i<matrix_size*matrix_size; i++){
           if(std::abs(h_check[i] - h_C[i])>tol){
                printf("result at index %d does not match, expected %f, got %f, abs difference is %f \n", i, h_check[i], h_C[i], std::abs(h_check[i] - h_C[i]));
                return 1;
           }
        }
    printf("Passed!\n");


    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_check;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle); //free cuBLAS handle

    return 0;
}