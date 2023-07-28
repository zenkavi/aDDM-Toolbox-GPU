# aDDM-Toolbox-GPU #

This toolbox can be used to perform model fitting and data simulation for the Drift Diffusion Model (DDM) and the attentional Drift Diffusion Model (aDDM). It is aimed to provide computational speedup, employing GPU optimizations for parameter estimations. 

## Requirements ##

This library requires NVCC and CUDA Toolkit versions 12.2 and g++ version 11.3.0. This library also uses two thrid-party C++ packages for thread pools and JSON processing: 

* [BS::thread_pool](https://github.com/bshoshany/thread-pool)
* [JSON for Modern C++](https://github.com/nlohmann/json)


## Installation and Usage ## 

The aDDM-Toolbox-GPU library can be built and installed in one step: 

```shell
$ make install
```

This will install the libaddm.so package as well as the corresponding header files. Although there are multiple header files corresponding to the aDDM and DDM programs, simply adding `#include <addm/gpu_toolbox.cuh>` to a C++/CUDA program will include all necessary headers. A simple usage example is described below: 

`main.cpp`:
```C++
#include <addm/gpu_toolbox.cuh>
#include <iostream>

void main() {
    aDDM addm = aDDM(0.005, 0.07, 0.5, 1);
    std::cout << "d: " << addm.d << std::endl; 
    std::cout << "sigma: " << addm.sigma << std::endl; 
    std::cout << "theta: " << addm.theta << std::endl; 
}
```

When compiling any code using the toolbox, include the `-laddm` flag to link with the installed shared object library.

```shell
$ nvcc -o main main.cpp -laddm
$ ./main
d: 0.005
sigma: 0.07
theta: 0.5
```

## Useful Functions ##

The following methods are useful for data simulation and 



