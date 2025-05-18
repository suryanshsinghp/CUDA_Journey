#include <iostream>
#include <stdio.h>

__global__ void threadID_x(){

    printf("I am thread: %u launched from block: %u and my count is: %u \n", threadIdx.x,blockIdx.x, blockDim.x * blockIdx.x + threadIdx.x);
}

int main(){

    dim3 block_dim(5, 1, 1); // 5 threads in x dir
    dim3 grid_dim(2, 1, 1); // 2 blocks in x dir

    threadID_x<<<grid_dim,block_dim>>>();  // use 2 block and launch 5 thread from each block [x-dir]
    //we do not have any input
    cudaDeviceSynchronize();



    return 0;
}