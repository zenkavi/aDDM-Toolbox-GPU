#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdio.h>
#include "../include/ddm.cuh"
#include "../include/util.h"
#include <cuda.h>
#include <cuda_runtime.h>


float d = 0.005;
float sigma = 0.07;
int barrier = 1;
int valueLeft = 3;

int main() {
    std::vector<DDMTrial> hostTrials = DDMTrial::loadTrialsFromCSV("results/ddm_simulations.csv");
    std::cout << "Counted " << hostTrials.size() << " trials." << std::endl;

    DDM ddm = DDM(d, sigma, barrier);
    double NLL = ddm.computeGPUNLL(hostTrials);
    std::cout << "NLL " << NLL << std::endl; 
}