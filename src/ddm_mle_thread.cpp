#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <chrono>
#include <sstream>
#include <addm/gpu_toolbox.cuh>


using namespace std::chrono;

std::vector<float> rangeD = {0.004, 0.005, 0.006};
std::vector<float> rangeSigma = {0.05, 0.07, 0.09};

int barrier = 1;

int main() {
    std::vector<DDMTrial> trials = DDMTrial::loadTrialsFromCSV("results/ddm_simulations.csv"); 
    std::cout << "Counted " << trials.size() << " trials." << std::endl;
    MLEinfo<DDM> info = DDM::fitModelMLE(trials, rangeD, rangeSigma, barrier, "thread", true);
    std::cout << "Optimal d=" << info.optimal.d << " sigma=" << info.optimal.sigma << std::endl; 

    std::ofstream fp;
    fp.open("results/ddm_likelihoods.csv");
    fp << "d,sigma,p\n";
    for (auto i : info.likelihoods) {
        fp << i.first.d << "," << i.first.sigma << "," << i.second << "\n";
    }
    fp.close();

}