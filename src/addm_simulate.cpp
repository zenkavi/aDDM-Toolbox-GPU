#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <cassert>
#include "addm.cuh"
#include "ddm.cuh"
#include "util.h"

int N = 1000;
float d = 0.005;
float sigma = 0.07;
float theta = 0.5;
float barrier = 1.0;

std::vector<int> valDiffs = {-3, -2, -1, 0, 1, 2, 3};

int main() {
    std::vector<aDDMTrial> outputs;

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
        outputs.push_back(adt);
    }

    std::cout << "printing outputs..." << std::endl;
    std::ofstream fp;
    fp.open("results/addm_simulations.csv");
    fp << "ID,choice,RT,valDiff,fixItem,fixTime\n";
    int id = 0; 
    for (aDDMTrial adt : outputs) {
        assert(adt.fixItem.size() == adt.fixTime.size());
        for (int i = 0; i < adt.fixItem.size(); i++) {
            fp << id << "," << adt.choice << "," << adt.RT << "," << 
                adt.valueLeft - adt.valueRight << "," << 
                adt.fixItem[i] << "," << adt.fixTime[i] << "\n";
        }
        id++;
    }
    fp.close();

}