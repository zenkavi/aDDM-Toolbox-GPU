#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <string> 
#include <random>
#include <fstream>
#include "nlohmann/json.hpp"
#include "util.h"
#include "ddm.h"


DDMTrial::DDMTrial(unsigned int RT, int choice, int valueLeft, int valueRight) {
    this->RT = RT;
    this->choice = choice;
    this->valueLeft = valueLeft;
    this->valueRight = valueRight;
}

DDM::DDM(float d, float sigma, float barrier, unsigned int nonDecisionTime, float bias) {
    if (barrier <= 0) {
        throw std::invalid_argument("barrier parameter must be larger than 0.");
    }
    if (bias >= barrier) {
        throw std::invalid_argument("bias parameter must be smaller than barrier parameter.");
    }
    this->d = d;
    this->sigma = sigma; 
    this->barrier = barrier; 
    this->nonDecisionTime = nonDecisionTime;
    this->bias = bias;            
}

double DDM::getTrialLikelihood(DDMTrial trial, int timeStep, float approxStateStep) {
    std::cout << "RT " << trial.RT << std::endl;
    int numTimeSteps = trial.RT / timeStep;
    if (numTimeSteps < 1) {
        throw std::invalid_argument("trial response time is smaller than time step.");
    }
    std::vector<float> barrierUp(numTimeSteps);
    std::fill(barrierUp.begin(), barrierUp.end(), this->barrier);
    std::vector<float> barrierDown(numTimeSteps);
    std::fill(barrierDown.begin(), barrierDown.end(), -this->barrier);

    for (int i = 1; i < numTimeSteps; i++) {
        barrierUp.at(i) = this->barrier / (1 + (DECAY * i));
        barrierDown.at(i) = -this->barrier / (1 + (DECAY * i));
    }

    int halfNumStateBins = ceil(this->barrier / approxStateStep);
    float stateStep = this->barrier / (halfNumStateBins + 0.5);
    std::vector<float> states;
    for (float ss = barrierDown.at(0) + (stateStep / 2);  ss < barrierUp.at(0) - (stateStep / 2); ss += stateStep) {
        states.push_back(ss);
    }

    std::cout << "STATES" << std::endl;
    for (auto s : states) { std::cout << s << std::endl; }
    std::cout << "------" << std::endl;

    float biasStateVal = MAXFLOAT;
    int biasState = 0;
    for (int i = 0; i < states.size(); i++) {
        float r = abs(states.at(i) - this->bias);
        if (r < biasStateVal) {
            biasState = i;
            biasStateVal = r;
        }
    }
    
    std::vector<std::vector<double>> prStates; // prStates[state][timeStep]
    for (int i = 0; i < states.size(); i++) {
        prStates.push_back({});
        for (int j = 0; j < numTimeSteps; j++) {
            prStates.at(i).push_back(0);
        }
    }
    std::vector<double> probUpCrossing; 
    std::vector<double> probDownCrossing;
    for (int i = 0; i < numTimeSteps; i++) {
        probUpCrossing.push_back(0);
        probDownCrossing.push_back(0);
    }
    prStates.at(biasState).at(0) = 1; 

    std::vector<std::vector<float>> changeMatrix(states.size(), std::vector<float>(states.size())); 
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < states.size(); j++) {
            changeMatrix[i][j] = states[i] - states[j];
        }
    }

    for (auto i : changeMatrix) {
        for (float k : i) {
            std::string ks = std::to_string(k);
            ks += std::string(10 - ks.length(), ' ');
            std::cout << ks;
        }
        std::cout << std::endl;
    }

    std::vector<std::vector<float>> changeUp(states.size(), std::vector<float>(numTimeSteps));
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < numTimeSteps; j++) {
            changeUp[i][j] = barrierUp[j] - states[i];
        }
    }

    std::vector<std::vector<float>> changeDown(states.size(), std::vector<float>(numTimeSteps));
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < numTimeSteps; j++) {
            changeDown[i][j] = barrierDown[j] - states[i];
        }
    }

    int elapsedNDT = 0;
    for (int time = 1; time < numTimeSteps; time++) {
        float mean;
        if (elapsedNDT < this->nonDecisionTime / timeStep) {
            mean = 0;
            elapsedNDT += 1;
        } else {
            mean = this->d * (trial.valueLeft - trial.valueRight);
        }

        std::cout << "PROBABILITY CHANGE MATRIX" << std::endl;
        std::vector<std::vector<double>> probDistChangeMatrix(states.size(), std::vector<double>(states.size())); 
        for (size_t i = 0; i < states.size(); i++) {
            for (size_t j = 0; j < states.size(); j++) {
                float x = changeMatrix[i][j];
                probDistChangeMatrix[i][j] = probabilityDensityFunction(mean, this->sigma, x);
            }
        }
        std::vector<double> prTimeSlice(states.size());
        for (size_t i = 0; i < states.size(); i++) {
            prTimeSlice[i] = prStates[i][time - 1];
        }
        std::vector<double> probChangeDotProduct(states.size()); 
        for (size_t i = 0; i < states.size(); i++) {
            double row_sum = 0;
            for (size_t j = 0; j < states.size(); j++) {
                row_sum += probDistChangeMatrix[i][j] * prTimeSlice[j];
            }
            probChangeDotProduct[i] = stateStep * row_sum; 
        }



        for (auto i : probDistChangeMatrix) {
            for (float k : i) {
                std::string ks = std::to_string(k);
                ks += std::string(10 - ks.length(), ' ');
                std::cout << ks;
            }
            std::cout << std::endl;
        }

        std::cout << "probability vector" << std::endl;
        for (auto i : prTimeSlice) {
            std::cout << i << std::endl;
        }

        std::cout << "dot product" << std::endl;
        for (auto i : probChangeDotProduct) {
            std::cout << i << std::endl;
        }

        break;
    }


    return 0;
}

DDMTrial DDM::simulateTrial(int ValueLeft, int ValueRight, int timeStep) {
    float RDV = this->bias;
    int time = 0;
    int elapsedNDT = 0;
    int RT;
    int choice;
    float mean;
    std::vector<float>RDVs = {RDV};
    std::random_device rd;
    std::mt19937 gen(rd()); 

    while (true) {
        if (RDV >= this->barrier || RDV <= -this->barrier) {
            RT = time * timeStep;
            if (RDV >= this->barrier) {
                choice = -1;
            } else {
                choice = 1;
            }
            break;
        }
        if (elapsedNDT < this->nonDecisionTime / timeStep) {
            mean = 0;
            elapsedNDT += 1;
        }
        else {
            mean = this->d * (ValueLeft - ValueRight);
        }
        std::normal_distribution<float> dist(mean, this->sigma);
        float inc = dist(gen);
        RDV += inc;
        RDVs.push_back(RDV);
        time += 1;
    }
    DDMTrial trial = DDMTrial(RT, choice, ValueLeft, ValueRight);
    trial.RDVs = RDVs;
    trial.timeStep = timeStep;
    return trial;
}
