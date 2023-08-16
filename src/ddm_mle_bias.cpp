#include <iostream>
#include <vector> 
#include <fstream>
#include <sstream>
#include <addm/gpu_toolbox.h>


std::vector<float> rangeD = {0.004, 0.005, 0.006};
std::vector<float> rangeSigma = {0.05, 0.07, 0.09};
std::vector<float> rangeBias = {-0.12, 0, 0.12};

int barrier = 1;

int main() {
    std::vector<DDMTrial> trials = DDMTrial::loadTrialsFromCSV("results/ddm_simulations.csv"); 
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    MLEinfo<DDM> info = DDM::fitModelMLE(trials, rangeD, rangeSigma, "thread", false, 1, 0, rangeBias);

    std::cout << "Optimal d=" << info.optimal.d << 
                 " sigma=" << info.optimal.sigma << 
                 " bias=" << info.optimal.bias << std::endl; 

    std::ofstream fp;
    fp.open("results/ddm_likelihoods.csv");
    fp << "d,sigma,bias,p\n";
    for (auto i : info.likelihoods) {
        fp << i.first.d << "," << i.first.sigma << "," << i.first.bias << "," << i.second << "\n";
    }
    fp.close();

}