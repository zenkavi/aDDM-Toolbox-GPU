#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <map>
#include <string>
#include "addm.h"

// #ifndef __NVCC__
#include "bshoshany/BS_thread_pool.hpp"
// #endif

extern float SEED;
extern float DECAY;
extern vector<string> validFixDistTypes;

struct EXPEntry {
    int parcode;
    int trial;
    int rt;
    int choice;
    int item_left;
    int item_right;
    int valid;
};

struct FIXEntry {
    int parcode;
    int trial;
    int fix_item;
    int fix_time;
};

double probabilityDensityFunction(float mean, float sigma, float x);

double cumulativeDensityFunction(float mean, float sigma, float x);

std::map<int, std::vector<aDDMTrial>> loadDataFromCSV(
    std::string expDataFilename,
    std::string fixDataFilename);

FixationData getEmpiricalDistributions(
    std::map<int, std::vector<aDDMTrial>> data, 
    int timeStep=10, int MaxFixTime=3000,
    int numFixDists=3, std::string fixDistType="simple",
    std::vector<int> valueDiffs={-3,-2,-1,0,1,2,3},
    std::vector<int> subjectIDs={},
    bool useOddTrials=true, 
    bool useEvenTrials=true, 
    bool useCisTrials=true, 
    bool useTransTrials=true
    );

void DDMexportData(DDM ddm, DDMTrial dt);

void aDDMexportData(aDDM addm, aDDMTrial adt);

template <class T> 
void printMatrix(std::vector<std::vector<T>> mat, std::string name) {
    std::cout << name << std::endl;
    for (auto row : mat) {
        for (auto f : row) {
            std::cout << f;
            if (f >= 0 && f < 10) {
                std::cout << "  ";
            } else {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "------" << std::endl;    
}

#endif