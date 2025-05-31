//install NVIDIA HPC SDK
//nvc++ -acc -Minfo=accel -mp -o out l6_openACC_SAXPY.cpp && ./out
//nsys profile --stats=true -o prof ./out > l6_profile.txt to profile the performance
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define vec_size 1000000000
//#define GPUONLY

void saxpy_acc(int n, float a, float *x, float *y) {
    //#pragma acc set device_type(nvidia)
    #pragma acc data copyin(x[0:n], a) copy(y[0:n])
{
    #pragma acc parallel loop
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}
}

void saxpy(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy_openmp(int n, float a, float *x, float *y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

int main(){
    float a = 2.0f;
    float *x = new float[vec_size];
    float *y = new float[vec_size];
    float *y_gpu = new float[vec_size];
    float *y_openmp = new float[vec_size];
    for (int i=0; i<vec_size; i++){
        x[i] = static_cast<float>(i);
        //similar to LAPACK saxpy, y is overwritten with result, so we have three y below
        y[i] = static_cast<float>(i);
        y_gpu[i] = y[i]; // because y* will be overwritten
        y_openmp[i] = y[i];
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    #ifndef GPUONLY
    saxpy(vec_size, a, x, y);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "CPU time: " << elapsed.count() << " seconds" << std::endl;
    #endif

    #ifndef GPUONLY
    start = std::chrono::high_resolution_clock::now();
    saxpy_openmp(vec_size, a, x, y_openmp);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "OpenMP time: " << elapsed.count() << " seconds" << std::endl;
    #endif

    start = std::chrono::high_resolution_clock::now();
    saxpy_acc(vec_size, a, x, y_gpu);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "GPU time: " << elapsed.count() << " seconds" << std::endl;

    #ifndef GPUONLY
    //verify that y and y_gpu are same
    for (int i = 0; i < vec_size; i++) {
        if (y[i] != y_gpu[i] || y_openmp[i] != y_gpu[i]) {
            std::cout << "Mismatch at index " << i << ": CPU y = " << y[i] << ", GPU y = " << y_gpu[i] << ", OpenMP y = " << y_openmp[i] << std::endl;
            return 1;
        }
    }
    #endif

    delete[] x;
    delete[] y;
    delete[] y_gpu;
    delete[] y_openmp;
    std::cout << "Passed!" << std::endl;

    return 0;
}