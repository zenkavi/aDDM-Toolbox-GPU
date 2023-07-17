#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "../include/addm.h"
#include "../include/ddm.h"
#include "../include/util.h"


using namespace std::chrono;

std::vector<float> rangeD = {0.003, 0.004, 0.005, 0.006};
std::vector<float> rangeSigma = {0.06, 0.07, 0.08, 0.09};
std::vector<float> rangeTheta = {0.4, 0.5, 0.6, 0.7};

int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<aDDMTrial> trials; 
    std::vector<aDDM> addms;
    std::ifstream file("results/addm_simulations.csv");
    std::string line;
    std::getline(file, line);

    int ID;
    int choice; 
    int RT; 
    int valDiff;
    int prevID;
    int fItem;
    int fTime; 
    bool firstIter = true; 
    std::vector<int> fixItem;
    std::vector<int> fixTime; 

    aDDMTrial adt = aDDMTrial();
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::getline(ss, field, ',');
        ID = std::stoi(field);
        std::getline(ss, field, ',');
        choice = std::stoi(field);
        std::getline(ss, field, ',');
        RT = std::stoi(field);
        std::getline(ss, field, ',');
        valDiff = std::stoi(field);
        std::getline(ss, field, ',');
        fItem = std::stoi(field);
        std::getline(ss, field, ',');
        fTime = std::stoi(field);
        if (ID == prevID && !firstIter) {
            adt.fixItem.push_back(fItem);
            adt.fixTime.push_back(fTime);
        } else {
            if (firstIter) {
                firstIter = false; 
            } else {
                trials.push_back(adt);
            }
            adt = aDDMTrial(RT, choice, valueLeft, valueLeft - valDiff);
            adt.fixItem.push_back(fItem);
            adt.fixTime.push_back(fTime);

        }
        prevID = ID;
    }
    trials.push_back(adt);
    file.close();
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

    auto start = high_resolution_clock::now();

    for (aDDM addm : addms) {
        std::cout << "Testing d=" << addm.d << " sigma=" << addm.sigma << " theta=" << addm.theta << std::endl;
        double NLL = aDDMParallelNLL(addm, trials);
        std::cout << "NLL=" << NLL << std::endl;
        if (NLL < minNLL) {
            minNLL = NLL;
            minD = addm.d;
            minSigma = addm.sigma;
            minTheta = addm.theta;
        }
        fp << addm.d << "," << addm.sigma << "," << addm.theta << "," << NLL << "\n";
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    fp.close();

    std::cout << 
    "  Optimal Parameters  \n" << 
    "======================\n" <<
    "d      : " << minD << "\n" << 
    "sigma  : " << minSigma << "\n" << 
    "theta  : " << minTheta << "\n" << 
    "NLL    : " << minNLL << "\n" <<
    "time   : " << duration.count() << std::endl;
}