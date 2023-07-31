#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <chrono>
#include "../include/ddm.cuh"
#include "util.h"

float d = 0.005;
float sigma = 0.07;
int barrier = 1;
int valueLeft = 3; 

using namespace std::chrono;

int main() {
    std::vector<DDMTrial> trials = DDMTrial::loadTrialsFromCSV("results/ddm_simulations.csv");
    std::cout << "counted " << trials.size() << std::endl;
    DDM ddm = DDM(d, sigma, barrier);

    std::ofstream fp;
    fp.open("results/ddm_simulations_prob.csv");
    fp << "choice,RT,p\n";

    auto start = high_resolution_clock::now();
    double NLL = 0; 
    for (DDMTrial dt : trials) {
        double prob = ddm.getTrialLikelihood(dt);
        fp << dt.choice << "," << dt.RT << "," << prob << "\n";
        NLL += -log(prob);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    fp.close();
    std::cout << "NLL: " << NLL << std::endl;
    std::cout << "time: " << duration.count() << " ms" << std::endl; 
}