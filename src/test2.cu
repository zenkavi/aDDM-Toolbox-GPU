#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <cublas_v2.h>
#include <cuda.h>
#include "../include/ddm.h"
#include "../include/util.h"

// std::vector<float> rangeD = {0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009};
// std::vector<float> rangeSigma = {0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};

std::vector<float> rangeD = {0.003};
std::vector<float> rangeSigma = {0.03};

int barrier = 1;
int valueLeft = 3; 

// CUDA kernel function to calculate likelihoods for a single DDM and DDMTrial combination
__global__ void calculateLikelihoods(const DDM* ddms, const DDMTrial* trials, double* likelihoods, int numDDMs, int numTrials) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;  
    int timeStep = 10;
    int approxStateStep = 0.1; 
    
    if (index >= numDDMs) {
        return; 
    }
    int ddmIndex = index;
    DDM ddm = ddms[ddmIndex];

    int sum = 0;
    for (int i = 0; i < numTrials; i++) {
        DDMTrial trial = trials[i];

        int RT; 
        RT = trial.RT; 
        
        sum += RT;
    }
    likelihoods[index] = sum;
        
    
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<DDMTrial> trials;
    std::vector<DDM> ddms;
    std::ifstream file("results/ddm_simulations.csv");
    std::string line;
    std::getline(file, line);
    int choice;
    int RT;
    int valDiff;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::getline(ss, field, ',');
        choice = std::stoi(field);
        std::getline(ss, field, ',');
        RT = std::stoi(field);
        std::getline(ss, field, ',');
        valDiff = std::stoi(field);
        DDMTrial dt = DDMTrial(RT, choice, valueLeft, valueLeft - valDiff);
        trials.push_back(dt);
    }
    file.close();
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    for (float d : rangeD) {
        for (float sigma : rangeSigma) {
            ddms.push_back(DDM(d, sigma, barrier));
        }
    }

    // Prepare data for GPU
    size_t numDDMs = ddms.size();
    size_t numTrials = trials.size();
    size_t numCombinations = numDDMs * numTrials;

    DDM* gpuDDMs;
    DDMTrial* gpuTrials;
    double* gpuLikelihoods;
    cudaMalloc(&gpuDDMs, numDDMs * sizeof(DDM));
    cudaMalloc(&gpuTrials, numTrials * sizeof(DDMTrial));
    cudaMalloc(&gpuLikelihoods, numDDMs * sizeof(double));

    // Transfer data from CPU to GPU
    cudaMemcpy(gpuDDMs, ddms.data(), numDDMs * sizeof(DDM), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuTrials, trials.data(), numTrials * sizeof(DDMTrial), cudaMemcpyHostToDevice);


    // Launch the GPU kernel
    int blockSize = 256;
    int numBlocks = (numDDMs + blockSize - 1) / blockSize;
    calculateLikelihoods<<<numBlocks, blockSize>>>(gpuDDMs, gpuTrials, gpuLikelihoods, numDDMs, numTrials);

    // Retrieve results from GPU
    std::vector<double> likelihoods(numTrials);
    cudaMemcpy(likelihoods.data(), gpuLikelihoods, numDDMs * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "likelihoods size: " << likelihoods.size() << std::endl;
    std::cout << "combinations    : " << rangeD.size() * rangeSigma.size() << std::endl;
    
    for (double l : likelihoods) {
        std::cout << l << std::endl;
    }

    // Clean up GPU resources
    cudaFree(gpuDDMs);
    cudaFree(gpuTrials);
    cudaFree(gpuLikelihoods);

    return 0;
}