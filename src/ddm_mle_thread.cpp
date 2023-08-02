#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <chrono>
#include <sstream>
#include <addm/gpu_toolbox.cuh>


using namespace std::chrono;

std::vector<float> rangeD = {0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009};
std::vector<float> rangeSigma = {0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};

int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<DDMTrial> trials = DDMTrial::loadTrialsFromCSV("results/ddm_simulations.csv"); 
    std::cout << "Counted " << trials.size() << " trials." << std::endl;
    MLEinfo<DDM> info = DDM::fitModelMLE(trials, rangeD, rangeSigma, barrier, "gpu", true);
    std::cout << "Optimal d=" << info.optimal.d << " sigma=" << info.optimal.sigma << std::endl; 

    std::ofstream fp;
    fp.open("results/ddm_likelihoods.csv");
    fp << "d,sigma,p\n";
    for (auto i : info.likelihoods) {
        fp << i.first.d << "," << i.first.sigma << "," << i.second << "\n";
    }
    fp.close();

}