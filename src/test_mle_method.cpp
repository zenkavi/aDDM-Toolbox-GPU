#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "../include/ddm.cuh"
#include "../include/addm.cuh"
#include "../include/util.h"

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

    MLEinfo<DDM> info = DDM::fitModelMLE(trials, rangeD, rangeSigma, 1, "gpu");
    DDM optimal = info.optimal; 

    std::cout << optimal.d << std::endl; 
    std::cout << optimal.sigma << std::endl; 

    double sum = 0; 
    for (auto i : info.posteriors) {
        std::cout << i.first.d << " " << i.first.sigma << " " << i.second << std::endl;
        sum += i.second; 
    }
    std::cout << "sum " << sum << std::endl; 
}