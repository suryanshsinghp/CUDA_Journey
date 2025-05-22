# FILES:

| **FILENAME**                       | **DESCRIPTION**                                              |
| ---------------------------------- | ------------------------------------------------------------ |
| `l1_Hello_from_thread.cu`          | Hello World equivalent for CUDA                              |
| `l2_vector_addition.cu`            | Vector addition with CUDA error checking <br /> 1D threads and blocks |
| `l3_naive_matmul_and_profiling.cu` | Matrix multiplication with profiling using CUDA events and also `nvtx` so saved profile can be opened in Nsight compute.`nvtx` inside preprocessor directives to avoid computational overhead when not profiling. |
| `l4_matmul_using_cuBLAS.cu`        | matrix multiplication using cuBLAS. NOTES: cuBLAS is similar to BLAS & Fortran and is column major. The order of matrix need to be switched because $(B^TA^T)^T=(A^T)^T(B^T)^T=AB$. |
| `l5_array_sum_atomics.cu`          | atomics to get sum of array. Despite being memory safe, atomics is serial and should be combined with other techniques such as reduction (next part). |
|                                    |                                                              |
|                                    |                                                              |
|                                    |                                                              |
|                                    |                                                              |

# TODOs:

Shared memory, atomics, wraps, streams, cutlass, cuDNN etc.

# Resources:

[Nvidia CUDA Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

[Nvidia cuBLAS Doc](https://docs.nvidia.com/cuda/cublas/)

Tutorials by NVIDIA & Oak Ridge National Laboratory
