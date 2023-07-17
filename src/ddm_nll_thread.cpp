#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "../include/ddm.h"
#include "../include/util.h"
#include "../include/bshoshany/BS_thread_pool.hpp"
#include <chrono>

using namespace std::chrono;

float d = 0.005;
float sigma = 0.07;
int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<DDMTrial> trials; 
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
    
    DDM ddm = DDM(d, sigma, barrier);

    auto start = high_resolution_clock::now(); 
    double NLL = DDMparallelNLL(ddm, trials);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Duration: " << duration.count() << " ms" << std::endl;
    std::cout << "NLL: " << NLL << std::endl;
}