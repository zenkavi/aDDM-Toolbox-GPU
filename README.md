# aDDM-Toolbox-GPU #

This toolbox can be used to perform model fitting and data simulation for the Drift Diffusion Model (DDM) and the attentional Drift Diffusion Model (aDDM). It is aimed to provide computational speedup, employing GPU optimizations for parameter estimations. 

## Requirements ##

This library requires NVCC and CUDA Toolkit versions 12.2 and g++ version 11.3.0. This library also uses two thrid-party C++ packages for thread pools and JSON processing: 

* [BS::thread_pool](https://github.com/bshoshany/thread-pool)
* [JSON for Modern C++](https://github.com/nlohmann/json)

These dependencies can be installed using the following commands: 

```shell
$ wget -O /usr/include/c++/11/BS_thread_pool.hpp https://raw.githubusercontent.com/bshoshany/thread-pool/master/include/BS_thread_pool.hpp
$ wget -O /usr/include/c++/11/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
```

*Note that the installation directory /usr/include/c++/11 may be modified to support newer versions of C++.*


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

## Examples ##

For any data simulation and model fitting, the DDM and aDDM classes are going to be the most useful. The `addm.cuh` amd `ddm.cuh` files can be found in the include directory. Below are some examples that may give you an idea how to use these classes. Besides these examples, you may want to:

* Check out the [documentation](https://jakegoldm.github.io/aDDM-Toolbox-GPU/).
* See example use cases in the [source directory](src/)

### N-Trial Simulation ###

Both the `ddm` and `addm` classes provide methods for simulating mass amounts of trials. This can be done using the `simulateTrial` method defined in both classes: 

#### DDM ####
```C++
#include <addm/gpu_toolbox.cuh>
#include <vector>

using namespace std; 

int N = 1000; 
int valueLeft = 3; 
int valueRight = 7; 

int main() {
    vector<DDMTrial> trials(N);
    DDM ddm = DDM(0.005, 0.07);

    for (int i = 0; i < N; i++) {
        DDMTrial trial = ddm.simulateTrial(valueLeft, ValueRight)
        trials.push_back(trial);
    }
}
```

#### aDDM #####

Note that for generating aDDM trials, an existing set of empirical fixations is required. This data can be packaged in the form of 1-2 CSVs. The input format for a single CSV is identical to the output of `aDDMTrial::writeTrialsToCSV(...)`. This CSV should be formatted as follows: 

|  ID 	|choice |   RT	|  valueLeft 	|  valueRight 	|  fixItem 	|  fixTime 	|
|:-:	|:-:	|:-:	|:-:	        |:-:	        |:-:	    |:-:	    |
|   0	|  1 	|  350 	|   3	        |   0           |   0	    |   200	    |
|   0	|   1	|  350 	|   3	        |   0	        |   1	    |   150	    |
|   1	|   -1	|  400 	|   4	        |  5            |   0	    |   300	    |
|   1	|   -1	|  400 	|   4	        |   5	        |   2	    |   100	    |

To load data using the single-CSV format: 

```C++
#include <addm/gpu_toolbox.cuh>
#include <vector>

using namespace std; 

int N = 1000; 
int valueLeft = 3; 
int valueRight = 7; 

int main() {
    map<int, vector<aDDMTrial>> data = loadDataFromSingleCSV("data/addm_sims.csv");
    FixationData fixationData = getEmpiricalDistributions(data);

    vector<DDMTrial> trials(N);
    aDDM addm = aDDM(0.005, 0.07, 0.5, 1);

    for (int i = 0; i < N; i++) {
        aDDMTrial trial = addm.simulateTrial(valueLeft, valueRight, fixationData);
        trials.push_back(trial);
    }
}
```

In the case of using two CSVs, such as with real experimental data, the first CSV should contain the experimental data including subject parcode, trial ID, RT, choice, and item values. The second CSV should contain fixation data pertaining to each trial. Sample CSV files in an acceptable format can be found in the data directory. To load data using the two-CSV format: 

```C++
#include <addm/gpu_toolbox.cuh>
#include <vector>

using namespace std; 

int N = 1000; 
int valueLeft = 3; 
int valueRight = 7; 

int main() {
    map<int, vector<aDDMTrial>> data = loadDataFromCSV("data/expdata.csv", "data/fixations.csv");
    FixationData fixationData = getEmpiricalDistributions(data);

    vector<DDMTrial> trials(N);
    aDDM addm = aDDM(0.005, 0.07, 0.5, 1);

    for (int i = 0; i < N; i++) {
        aDDMTrial trial = addm.simulateTrial(valueLeft, valueRight, fixationData);
        trials.push_back(trial);
    }
}
```
