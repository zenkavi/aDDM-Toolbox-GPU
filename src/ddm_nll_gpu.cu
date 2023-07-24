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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


float d = 0.005;
float sigma = 0.07;
int barrier = 1;
int valueLeft = 3;

int main() {
    std::vector<DDMTrial> hostTrials; 
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
        hostTrials.push_back(dt);
    }
    file.close();
    std::cout << "Counted " << hostTrials.size() << " trials." << std::endl;

    DDM ddm = DDM(d, sigma, barrier);
    double NLL = ddm.computeGPUNLL(hostTrials);
    std::cout << "NLL " << NLL << std::endl; 
}