#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <addm/gpu_toolbox.cuh>

std::vector<float> rangeD = {0.003, 0.004, 0.005, 0.006};
std::vector<float> rangeSigma = {0.06, 0.07, 0.08, 0.09};
std::vector<float> rangeTheta = {0.4, 0.5, 0.6, 0.7};

int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("results/addm_simulations.csv"); 
    std::vector<aDDM> addms;
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    for (float d : rangeD) {
        for (float sigma : rangeSigma) {
            for (float theta: rangeTheta) {
                addms.push_back(aDDM(d, sigma, theta, barrier));
            }
        }
    }

    std::ofstream fp;
    fp.open("results/addm_mle.csv");
    fp << "d,sigma,theta,NLL\n";

    double minNLL = __DBL_MAX__;
    double minD = 0;
    double minSigma = 0; 
    double minTheta = 0; 
    for (aDDM addm : addms) {
        std::cout << "Testing d=" << addm.d << " sigma=" << addm.sigma << " theta=" << addm.theta << std::endl;
        double NLL = 0; 
        for (aDDMTrial adt : trials) {
            double prob = addm.getTrialLikelihood(adt);
            NLL += -log(prob);
        }
        std::cout << "NLL=" << NLL << std::endl;
        if (NLL < minNLL) {
            minNLL = NLL;
            minD = addm.d;
            minSigma = addm.sigma;
            minTheta = addm.theta;
        }
        fp << addm.d << "," << addm.sigma << "," << addm.theta << "," << NLL << "\n";
    }
    fp.close();

    std::cout << 
    "  Optimal Parameters  \n" << 
    "======================\n" <<
    "d      : " << minD << "\n" << 
    "sigma  : " << minSigma << "\n" << 
    "theta  : " << minTheta << "\n" << 
    "NLL    : " << minNLL << std::endl;
}