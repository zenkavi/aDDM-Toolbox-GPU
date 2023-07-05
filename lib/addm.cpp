#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <map>
#include <ctime>
#include <time.h>
#include <cstdlib>
#include <random> 
#include "nlohmann/json.hpp"
#include "ddm.h"
#include "util.h"
#include "addm.h"


FixationData::FixationData(float probFixLeftFirst, std::vector<int> latencies, 
    std::vector<int> transitions, fixDists fixations, std::string fixDistType) {
    if (!std::count(
        validFixDistTypes.begin(), validFixDistTypes.end(), fixDistType)) {
            throw std::invalid_argument(
                "Argument type must be one of {simple, difficulty, fixation}");
        }
    this->probFixLeftFirst = probFixLeftFirst;
    this->latencies = latencies;
    this->transitions = transitions;
    this->fixations = fixations;
    this->fixDistType = fixDistType;
}

aDDMTrial::aDDMTrial(
    unsigned int RT, int choice, int valueLeft, int valueRight, 
    std::vector<int> fixItem, std::vector<int> fixTime, 
    std::vector<float> fixRDV, float uninterruptedLastFixTime) :
    DDMTrial(RT, choice, valueLeft, valueRight) {
        this->fixItem = fixItem;
        this->fixTime = fixTime;
        this->fixRDV = fixRDV;
        this->uninterruptedLastFixTime = uninterruptedLastFixTime;
}

aDDM::aDDM(float d, float sigma, float theta, float barrier, 
    unsigned int nonDecisionTime, float bias) : 
    DDM(d, sigma, barrier, nonDecisionTime, bias) {
        this->theta = theta;
}

aDDMTrial aDDM::simulateTrial(
    int valueLeft, int valueRight, FixationData fixationData, int timeStep, 
    int numFixDists, fixDists fixationDist, vector<int> timeBins) {

    srand(time(NULL));

    std::map<int, int> fixUnfixValueDiffs;
    fixUnfixValueDiffs.insert({1, valueLeft - valueRight});
    fixUnfixValueDiffs.insert({2, valueRight - valueLeft});
    std::vector<int> fixItem;
    std::vector<int> fixTime;
    std::vector<float> fixRDV;

    std::random_device rd;
    std::mt19937 gen(rd()); 

    float RDV = this->bias;
    int time = 0;
    int choice; 
    int uninterruptedLastFixTime;
    int RT;

    std::vector<float>RDVs = {RDV};

    std::uniform_int_distribution<std::size_t> ludist(0, fixationData.latencies.size() - 1);
    int rIDX = ludist(gen);
    int latency = fixationData.latencies.at(rIDX);
    int remainingNDT = this->nonDecisionTime - latency;

    // std::mt19937 gen(SEED);

    for (int t = 0; t < latency / timeStep; t++) {
        std::normal_distribution<float> ndist(0, this->sigma);
        float inc = ndist(gen);
        RDV += inc;
        RDVs.push_back(RDV);

        if(RDV >= this->barrier || RDV <= -this->barrier) {
            if (RDV >= this->barrier) {
                choice = -1;
            } else {
                choice = 1; 
            }
            fixRDV.push_back(RDV);
            fixItem.push_back(0);
            int dt = (t + 1) * timeStep;
            fixTime.push_back(dt);
            time += dt;
            RT = time;
            uninterruptedLastFixTime = latency;
            return aDDMTrial(
                RT, choice, valueLeft, valueRight, 
                fixItem, fixTime, fixRDV, uninterruptedLastFixTime);
        }
    }

    fixRDV.push_back(RDV);
    RDVs.push_back(RDV);
    fixItem.push_back(0);
    int dt = latency - (latency % timeStep);
    fixTime.push_back(dt);
    time += dt;

    int fixNumber = 1;
    int prevFixatedItem = -1;
    int currFixLocation = 0;
    bool decisionReached = false;
    float currFixTime;

    while (true) {
        if (currFixLocation == 0) {
            if (prevFixatedItem == -1) {
                std::discrete_distribution<> ddist({fixationData.probFixLeftFirst, 1 - fixationData.probFixLeftFirst});
                currFixLocation = ddist(gen) + 1;
            } else if (prevFixatedItem == 1) {
                currFixLocation = 2;
            } else if (prevFixatedItem == 2) {
                currFixLocation = 1;
            }
            prevFixatedItem = currFixLocation;
            // ASSUMING WE ARE USING FIXATION DATA DIST FOR NOW
            if (fixationDist.empty()) {
                // ASSUMING SIMPLE
                if (fixationData.fixDistType == "simple") {
                    vector<float> fixTimes = fixationData.fixations.at(fixNumber);
                    std::uniform_int_distribution<std::size_t> fudist(0, fixTimes.size() - 1);
                    rIDX = fudist(gen);
                    currFixTime = fixTimes.at(rIDX);
                } else {
                    throw std::invalid_argument("not implemented");
                }
            }
            if (fixNumber < numFixDists) {
                fixNumber++;
            }
        }
        else {
            currFixLocation = 0;
            rIDX = rand() % fixationData.transitions.size();
            currFixTime = fixationData.transitions.at(rIDX);
        }
        if (remainingNDT > 0)  {
            for (int t = 0; t < remainingNDT / timeStep; t++) {
                std::normal_distribution<float> ndist(0, this->sigma);
                float inc = ndist(gen);
                RDV += inc;
                RDVs.push_back(RDV);

                if(RDV >= this->barrier || RDV <= -this->barrier) {
                    if (RDV >= this->barrier) {
                        choice = -1;
                    } else {
                        choice = 1; 
                    }
                    fixRDV.push_back(RDV);
                    fixItem.push_back(currFixLocation);
                    int dt = (t + 1) * timeStep;
                    fixTime.push_back(dt);
                    time += dt;
                    RT = time;
                    uninterruptedLastFixTime = currFixTime;
                    decisionReached = true;
                    break;
                }
            }
        }
        if (decisionReached) {
            break;
        }
        float remainingFixTime = max(0.0f, currFixTime - max(0, remainingNDT));
        remainingNDT -= currFixTime;

        for (int t = 0; t < round(remainingFixTime / timeStep); t++) {
            float mean;
            if (currFixLocation == 0) {
                mean = 0;
            } else if (currFixLocation == 1) {
                mean = this->d * (valueLeft - (this->theta * valueRight));
            } else if (currFixLocation == 2) {
                mean = this->d * ((this->theta * valueLeft) - valueRight);
            }
            std::normal_distribution<float> ndist(mean, this->sigma);
            float inc = ndist(gen);
            RDV += inc;
            RDVs.push_back(RDV); 

            if(RDV >= this->barrier || RDV <= -this->barrier) {
                if (RDV >= this->barrier) {
                    choice = -1;
                } else {
                    choice = 1; 
                }
                fixRDV.push_back(RDV);
                fixItem.push_back(currFixLocation);
                int dt = (t + 1) * timeStep;
                fixTime.push_back(dt);
                time += dt;
                RT = time;
                uninterruptedLastFixTime = currFixTime;
                decisionReached = true;
                break;
            }                
        }

        if (decisionReached) {
            break;
        }

        fixRDV.push_back(RDV);
        fixItem.push_back(currFixLocation);
        int cft = round(currFixTime);
        int dt = cft - (cft % timeStep);
        fixTime.push_back(dt);
        time += dt;
    } 

    aDDMTrial trial = aDDMTrial(RT, choice, valueLeft, valueRight, fixItem, fixTime, fixRDV, uninterruptedLastFixTime);
    trial.RDVs = RDVs;
    trial.timeStep = timeStep;
    return trial;
}