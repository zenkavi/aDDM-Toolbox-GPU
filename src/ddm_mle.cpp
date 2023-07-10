#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "ddm.h"
#include "util.h"

std::vector<float> rangeD = {0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009};
std::vector<float> rangeSigma = {0.05, 0.06, 0.07, 0.08, 0.09};

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

    for (float d : rangeD) {
        for (float sigma : rangeSigma) {
            ddms.push_back(DDM(d, sigma, barrier));
        }
    }

    double minNLL = __DBL_MAX__;
    double minD = 0; 
    double minSigma = 0; 
    std::cout << "log " << log(0) << std::endl;
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
    }
    std::cout << 
    "  Optimal Parameters  \n" << 
    "======================\n" <<
    "d      : " << minD << "\n" << 
    "sigma  : " << minSigma << "\n" << 
    "NLL    : " << minNLL << std::endl;
}