//nvc++ -acc -Minfo=accel -mp -o out openACC_cuda.cpp && ./out > out_openACC_cuda.txt
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
#define beta ((alpha * dt) / (dx * dx)) // stability condition
#define flatIdx(x, y, width) ((y) * (width) + (x))
#define NO_FILE_WRITE //dont write file when N is large, just for testing


void updateT(float *T, float *T_new, int width, int iter) { //need to do iteration here for gpu
    #pragma acc data copyin(T[0:width*width]) copy(T_new[0:width*width])
    for (int t = 0; t < iter; ++t){
        #pragma acc parallel loop collapse(2) present(T, T_new)
    for (int j = 1; j < width -1; ++j) {
        for (int i = 1; i < width - 1; ++i) { //exclude boundaries
            T_new[flatIdx(i, j, width)] = T[flatIdx(i    , j    , width)] +
                                  beta * (T[flatIdx(i + 1, j    , width)] + 
                                          T[flatIdx(i - 1, j    , width)] +
                                          T[flatIdx(i    , j + 1, width)] + 
                                          T[flatIdx(i    , j - 1, width)] -
                                      4 * T[flatIdx(i    , j    , width)]);
        }
    }
    #pragma acc parallel loop present(T, T_new)
    for (int idx = 0; idx < width * width; ++idx) {
        T[idx] = T_new[idx];
    }
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
    std::string filename;

    auto start = std::chrono::high_resolution_clock::now();

    T = new float[N * N];
    T_new = new float[N * N];
    
    for (int i = 0; i < N * N; ++i) { //init with 0s
        T[i] = 0.0f;
    }
    
    for (int j = 0; j < N; ++j) {
        T[flatIdx(0, j, N)] = 100.0f; //left boundary at 100 degrees
    }
    std::memcpy(T_new, T, N * N * sizeof(float)); //starting point for T_new

    
    if (matrixToFile(T, N, "initial_T.txt") != 0) { //just to test if files are written correctly
        std::cerr << "Error writing initial state to file." << std::endl;
        return 1;
    }

    int num_steps = static_cast<int>(t_final / dt);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    start = end; //reset start for next timing
    std::cout << "Starting T updates, initialization and allocation took " << duration.count()/ 1000.0 << " seconds." << std::endl;


    updateT(T, T_new, N,num_steps);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Completed " << num_steps << " updates in " << duration.count()/ 1000.0 << " seconds." << std::endl;
    //write to file for final state
    filename = "T_"+std::to_string(num_steps)+".txt";
    if (matrixToFile(T_new, N, filename) != 0) {
        std::cerr << "Error writing final state to file." << std::endl;
        return 1;
    }

        delete[] T;
        delete[] T_new;
    std::cout << "Completed!" << std::endl;

    auto global_end = std::chrono::high_resolution_clock::now();
    auto global_duration = std::chrono::duration_cast<std::chrono::milliseconds>(global_end - global_start);
    std::cout << "Total execution time: " << global_duration.count() / 1000.0 << " seconds." << std::endl;
    return 0;
}