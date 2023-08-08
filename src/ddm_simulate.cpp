#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <addm/gpu_toolbox.h>

int N = 1000;
float d = 0.005;
float sigma = 0.07;
float barrier = 1.0;

std::vector<int> valDiffs = {-3, -2, -1, 0, 1, 2, 3};

int main() {

    std::vector<DDMTrial> trials;

    srand(time(NULL));

    std::cout << "reading data..." << std::endl;
    DDM ddm = DDM(d, sigma, barrier);

    for (int i = 0; i < N; i++) {
        if (i % 50 == 0) {
            std::cout << "performing trial " << i << "..." << std::endl;
        }

        std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<std::size_t> distribution(0, valDiffs.size() - 1);
        int rIDX = distribution(generator);

        int valDiff = valDiffs.at(rIDX);
        int valueLeft = 3;
        int valueRight = valueLeft - valDiff;
        DDMTrial dt = ddm.simulateTrial(valueLeft, valueRight);
        trials.push_back(dt);
    }
    DDMTrial::writeTrialsToCSV(trials, "results/ddm_simulations.csv");

}