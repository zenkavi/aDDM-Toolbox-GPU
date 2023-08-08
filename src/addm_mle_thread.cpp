#include <iostream>
#include <vector> 
#include <random>
#include <fstream>
#include <ctime>
#include <addm/gpu_toolbox.h>


using namespace std::chrono;

std::vector<float> rangeD = {0.00475, 0.005, 0.00525, 0.0055, 0.00575};
std::vector<float> rangeSigma = {0.065, 0.0675, 0.07, 0.0725, 0.075};
std::vector<float> rangeTheta = {0.475, 0.5, 0.525, 0.55, 0.575};

int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("results/addm_simulations.csv");
    auto start = high_resolution_clock::now(); 
    MLEinfo info = aDDM::fitModelMLE(trials, rangeD, rangeSigma, rangeTheta, barrier, "thread", true);
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<seconds>(stop - start);
    std::cout << 
    "  Optimal Parameters  \n" << 
    "======================\n" <<
    "d      : " << info.optimal.d << "\n" << 
    "sigma  : " << info.optimal.sigma << "\n" << 
    "theta  : " << info.optimal.theta << "\n" << 
    "time   : " << duration.count() << std::endl;

    std::ofstream fp; 
    fp.open("results/addm_posteriors.csv"); 
    fp << "d,sigma,theta,p" << std::endl; 
    for (auto &i : info.likelihoods) {
        fp << i.first.d << "," << i.first.sigma << "," << i.first.theta << "," << i.second << std::endl; 
    }
}