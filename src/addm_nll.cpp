#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <addm/gpu_toolbox.h>

float d = 0.005;
float sigma = 0.07;
float theta = 0.5;
int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("results/addm_simulations.csv");
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    aDDM addm = aDDM(d, sigma, theta, barrier);

    std::ofstream fp;
    fp.open("results/addm_simulations_prob.csv");
    fp << "choice,RT,p\n";

    double NLL = 0; 
    for (aDDMTrial adt : trials) {
        double prob = addm.getTrialLikelihood(adt);
        fp << adt.choice << "," << adt.RT << "," << prob << "\n";
        NLL += -log(prob);
    }
    fp.close();
    std::cout << "NLL: " << NLL << std::endl;
}