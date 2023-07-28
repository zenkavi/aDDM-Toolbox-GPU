#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "../include/addm.cuh"
#include "util.h"

std::vector<float> rangeD = {0.003, 0.004, 0.005};
std::vector<float> rangeSigma = {0.07, 0.08, 0.09};

int barrier = 1;
int valueLeft = 3; 

int main() {
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

    DDM optimal = DDM::fitModelBasic(trials, rangeD, rangeSigma, 1, "gpu");

    // DDM optimal = DDM::fitModelOptimized(trials, 0.1, 1, 0.05, 0.5, 1, 1, "gpu");

    std::cout << optimal.d << std::endl; 
    std::cout << optimal.sigma << std::endl; 
}