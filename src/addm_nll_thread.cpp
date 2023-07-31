#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <chrono>
#include <fstream>
#include <sstream>
#include "../include/addm.cuh"
#include "../include/ddm.cuh"
#include "../include/util.h"

using namespace std::chrono;

float d = 0.005;
float sigma = 0.07;
float theta = 0.5;

int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("results/addm_simulations.csv");
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    aDDM addm = aDDM(d, sigma, theta, barrier);

    auto start = high_resolution_clock::now(); 
    double NLL = addm.computeParallelNLL(trials);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Duration: " << duration.count() << " ms" << std::endl;
    std::cout << "NLL: " << NLL << std::endl;
}