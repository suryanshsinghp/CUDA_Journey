

Transient heat equation for conduction is given by: $\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2T}{\partial x^2}+\frac{\partial^2T}{\partial y^2} \right)$ where T is temperature, x and y are grid (assumed uniform), t is time and $\alpha$ is thermal diffusivity. The equation is discretized using forward Euler for time and central difference scheme for space and rearranging, we get the matrix update formula:

$T_{i,j}^{n+1}=(1-4\beta)T_{i,j}^n+\beta\left(T_{i+1,j}^n+T_{i-1,j}^n+T_{i,j+1}^n+T_{i,j-1}^n \right)$; where $\beta = \frac{\alpha\Delta t}{\Delta x^2}$



All results shown in the table below are based on grid size of 10,000 $\times$ 10,000 and 1000 timestep updates (or matrix updates).

| **Code**           | **Description**                                              | **matrix update time [s]** | **matrix update time scaled** | **Total time [s]** |
| ------------------ | ------------------------------------------------------------ | -------------------------- | ----------------------------- | ------------------ |
| `cpu.cpp`          | baseline implementation on CPU                               | 302.4                      | 504                           | 302.8              |
| `openMP.cpp`       | uses openMP to parallelize for loops on CPU                  | 22.3                       | 37.1                          | 22.5               |
| `openACC_cuda.cpp` | uses openACC pragma annotations for naive parallelize using CUDA compatible GPU. Useful for quickly making existing code parallel. | 4.3                        | 7.1                           | 4.7                |
| `cuda.cu`          | written using CUDA only for GPU                              | 0.6                        | 1                             | 1.2                |

