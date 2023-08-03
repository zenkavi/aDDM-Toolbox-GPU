#include <iostream>
#include <vector> 
#include <addm/gpu_toolbox.cuh>

float d = 0.005;
float sigma = 0.07;
float theta = 0.5;
int barrier = 1;

int main() {
    std::vector<aDDMTrial> trials = aDDMTrial::loadTrialsFromCSV("results/addm_simulations.csv");
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    aDDM addm = aDDM(d, sigma, theta, barrier);
    ProbabilityData data = addm.computeGPUNLL(trials);
    std::cout << "NLL:        " << data.NLL << std::endl;
    std::cout << "Likelihood: " << data.likelihood << std::endl; 
}