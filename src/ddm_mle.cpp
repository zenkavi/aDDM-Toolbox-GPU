#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <addm/gpu_toolbox.h>

std::vector<float> rangeD = {0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009};
std::vector<float> rangeSigma = {0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};

int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<DDMTrial> trials = DDMTrial::loadTrialsFromCSV("results/ddm_simulations.csv");
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    std::vector<DDM> ddms; 
    for (float d : rangeD) {
        for (float sigma : rangeSigma) {
            ddms.push_back(DDM(d, sigma, barrier));
        }
    }

    std::ofstream fp;
    fp.open("results/ddm_mle.csv");
    fp << "d,sigma,NLL\n";

    double minNLL = __DBL_MAX__;
    double minD = 0; 
    double minSigma = 0; 
    for (DDM ddm : ddms) {
        std::cout << "Testing combination d=" << ddm.d << " sigma=" << ddm.sigma << std::endl;
        double NLL = 0;
        for (DDMTrial dt : trials) {
            double prob = ddm.getTrialLikelihood(dt);
            NLL += -log(prob);
        }            
        std::cout << "NLL=" << NLL << std::endl;
        if (NLL < minNLL) {
            minNLL = NLL;
            minD = ddm.d;
            minSigma = ddm.sigma;
        }
        fp << ddm.d << "," << ddm.sigma << "," << NLL << "\n";
    }
    fp.close(); 

    std::cout << 
    "  Optimal Parameters  \n" << 
    "======================\n" <<
    "d      : " << minD << "\n" << 
    "sigma  : " << minSigma << "\n" << 
    "NLL    : " << minNLL << std::endl;
}