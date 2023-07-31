#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "../include/ddm.cuh"
#include "../include/util.h"
#include <chrono>

using namespace std::chrono;

float d = 0.005;
float sigma = 0.07;
int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<DDMTrial> trials = DDMTrial::loadTrialsFromCSV("results/ddm_simulations.csv");
    std::cout << "Counted " << trials.size() << " trials." << std::endl;
    
    DDM ddm = DDM(d, sigma, barrier);

    auto start = high_resolution_clock::now(); 
    double NLL = ddm.computeParallelNLL(trials);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Duration: " << duration.count() << " ms" << std::endl;
    std::cout << "NLL: " << NLL << std::endl;
}