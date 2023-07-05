#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include "addm.h"
#include "ddm.h"
#include "util.h"

int N = 1000;
float d = 0.005;
float sigma = 0.07;
float theta = 0.5;
float barrier = 1.0;

std::vector<int> valDiffs = {-3, -2, -1, 0, 1, 2, 3};

struct trialOutput {
    int choice;
    int RT;
    int valDiff;
};

int main() {

    std::vector<trialOutput> outputs;

    srand(time(NULL));

    std::cout << "reading data..." << std::endl;
    std::map<int, std::vector<aDDMTrial>> data = loadDataFromCSV("data/expdata.csv", "data/fixations.csv");
    FixationData fixationData = getEmpiricalDistributions(data);
    aDDM addm = aDDM(d, sigma, theta, barrier);

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
        aDDMTrial adt = addm.simulateTrial(valueLeft, valueRight, fixationData);

        trialOutput t; 
        t.choice = adt.choice;
        t.RT = adt.RT;
        t.valDiff = valDiff;

        outputs.push_back(t);
    }

    std::cout << "printing outputs..." << std::endl;
    std::ofstream fp;
    fp.open("results/simulations.csv");
    fp << "choice,RT,valDiff\n";
    for (trialOutput t : outputs) {
        fp << t.choice << "," << t.RT << "," << t.valDiff << "\n";
    }
    fp.close();

}