//nvcc -o out cuda.cu && ./out > out_cuda.txt
#include <iostream>
#include <stdlib.h> 
#include <fstream>
#include <string> //for file names
#include <cstring> // for memcpy
#include <iomanip>  // for std::setw
#include <chrono>



#define N 10000
#define t_final 1.0f
#define dt 0.001f
#define dx 0.01f //=dy
#define alpha 0.01f // thermal diffusivity
#define beta alpha * dt / (dx * dx) // stability condition
#define flatIdx(x, y, width) ((y) * (width) + (x))
#define NO_FILE_WRITE //dont write file when N is large, just for testing

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


__global__ void updateT(float *T, float *T_new, int width, int iter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    for (int t = 0; t < iter; ++t){
    if (idx > 0 && idx < width - 1 && idy > 0 && idy < width - 1) { //exclude boundaries
        T_new[flatIdx(idx, idy, width)] = T[flatIdx(idx, idy, width)] +
                            beta * (T[flatIdx(idx + 1, idy, width)] + 
                                    T[flatIdx(idx - 1, idy, width)] +
                                    T[flatIdx(idx, idy + 1, width)] + 
                                    T[flatIdx(idx, idy - 1, width)] -
                                4 * T[flatIdx(idx, idy, width)]);
    }

    __syncthreads();
    if (idx > 0 && idx < width - 1 && idy > 0 && idy < width - 1) { //exclude boundaries
        T[flatIdx(idx, idy, width)] = T_new[flatIdx(idx, idy, width)];
    }
    __syncthreads();
}
}

int matrixToFile(float *T, int width, const std::string &filename){
    #ifndef NO_FILE_WRITE
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return 1;
        }
        for (int j = 0; j < width; ++j) {
            for (int i = 0; i < width; ++i) {
                file << std::setw(8) <<std::fixed << std::setprecision(4) << T[flatIdx(i, j, width)] << " ";
            }
            file << std::endl;
        }
        file.close();
    #endif
    return 0; 
}

int main(){
    auto global_start = std::chrono::high_resolution_clock::now();
    float *T, *T_new;
    float *d_T, *d_T_new;
    std::string filename;

    auto start = std::chrono::high_resolution_clock::now();

    T = new float[N * N];
    T_new = new float[N * N];
    cudaMalloc(&d_T, N * N * sizeof(float));
    cudaMalloc(&d_T_new, N * N * sizeof(float));
    cudaCheckErrors("cudaMalloc");
    
    for (int i = 0; i < N * N; ++i) { //init with 0s
        T[i] = 0.0f;
    }
    
    for (int j = 0; j < N; ++j) {
        T[flatIdx(0, j, N)] = 100.0f; //left boundary at 100 degrees
    }
    std::memcpy(T_new, T, N * N * sizeof(float)); //starting point for T_new
    cudaMemcpy(d_T, T, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_new, T_new, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy");

    
    if (matrixToFile(T, N, "initial_T.txt") != 0) { //just to test if files are written correctly
        std::cerr << "Error writing initial state to file." << std::endl;
        return 1;
    }

    int num_steps = static_cast<int>(t_final / dt);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    start = end; //reset start for next timing
    std::cout << "Starting T updates, initialization and allocation took " << duration.count()/ 1000.0 << " seconds." << std::endl;

    int threads_per_block = 32; //cuda max is 1024
    int total_blocks = (N + threads_per_block - 1) / threads_per_block;
    //printf("Threads per block: %d, Total blocks: %d\n", threads_per_block, total_blocks);
    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(total_blocks, total_blocks);
    updateT<<<grid_dim,block_dim>>>(d_T, d_T_new, N, num_steps);
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel launch");

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Completed " << num_steps << " updates in " << duration.count()/ 1000.0 << " seconds." << std::endl;
    //write to file for final state
    cudaMemcpy(T_new, d_T_new, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy to host");
    filename = "T_"+std::to_string(num_steps)+".txt";
    if (matrixToFile(T_new, N, filename) != 0) {
        std::cerr << "Error writing final state to file." << std::endl;
        return 1;
    }

    delete[] T;
    delete[] T_new;
    cudaFree(d_T);
    cudaFree(d_T_new);
    std::cout << "Completed!" << std::endl;

    auto global_end = std::chrono::high_resolution_clock::now();
    auto global_duration = std::chrono::duration_cast<std::chrono::milliseconds>(global_end - global_start);
    std::cout << "Total execution time: " << global_duration.count() / 1000.0 << " seconds." << std::endl;
    return 0;
}