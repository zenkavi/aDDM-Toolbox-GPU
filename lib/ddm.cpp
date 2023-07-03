#include <stdexcept>
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

DDMTrial DDM::simulateTrial(int ValueLeft, int ValueRight, int timeStep) {
    float RDV = this->bias;
    int time = 0;
    int elapsedNDT = 0;
    int RT;
    int choice;
    float mean;
    std::vector<float>RDVs = {RDV};
    std::random_device rd;
    // std::mt19937 gen(rd()); 
    std::mt19937 gen(SEED);
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
