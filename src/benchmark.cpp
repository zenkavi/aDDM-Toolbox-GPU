#include <addm/gpu_toolbox.h>
#include <iostream>

using namespace std::chrono; 

std::vector<float> rangeD = {0.003, 0.005, 0.007};
std::vector<float> rangeSigma = {0.06, 0.07, 0.08};
std::vector<float> rangeTheta = {0.5, 0.6, 0.7};


int main() {
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("results/addm_simulations.csv");

    // auto start = high_resolution_clock::now(); 
    // MLEinfo info = aDDM::fitModelMLE(trials, rangeD, rangeSigma, rangeTheta, 1, "basic");
    // auto stop = high_resolution_clock::now(); 
    // auto duration_basic = duration_cast<seconds>(stop - start);

    // start = high_resolution_clock::now(); 
    // info = aDDM::fitModelMLE(trials, rangeD, rangeSigma, rangeTheta, 1, "thread");
    // stop = high_resolution_clock::now(); 
    // auto duration_thread = duration_cast<seconds>(stop - start);

    auto start = high_resolution_clock::now(); 
    MLEinfo info = aDDM::fitModelMLE(trials, rangeD, rangeSigma, rangeTheta, 1, "gpu");
    auto stop = high_resolution_clock::now(); 
    auto duration_gpu = duration_cast<seconds>(stop - start);

    // std::cout << "basic " << duration_basic.count() << std::endl; 
    // std::cout << "thread " << duration_thread.count() << std::endl; 
    std::cout << "gpu " << duration_gpu.count() << std::endl; 

}