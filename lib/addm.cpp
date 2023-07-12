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

double aDDM::getTrialLikelihood(aDDMTrial trial, bool debug, int timeStep, float approxStateStep) {
    if (debug) {
        std::cout << std::setprecision(6) << std::fixed;
    }
    std::vector<int> correctedFixItem = trial.fixItem;
    std::vector<int> correctedFixTime = trial.fixTime;
    if (this->nonDecisionTime > 0) {
        int remainingNDT = this->nonDecisionTime;
        assert(trial.fixItem.size() == trial.fixTime.size());
        for (int i = 0; i < trial.fixItem.size(); i++) {
            int fItem = trial.fixItem[i];
            int fTime = trial.fixTime[i];
            if (remainingNDT > 0) {
                correctedFixItem.push_back(0);
                correctedFixTime.push_back(min(remainingNDT, fTime));
                correctedFixItem.push_back(fItem);
                correctedFixTime.push_back(max(fTime - remainingNDT, 0));
            } else {
                correctedFixItem.push_back(fItem);
                correctedFixTime.push_back(fTime);
            }
        }
    }

    if (debug) {
        std::cout << "CFI" << std::endl;
        for (int i : correctedFixItem) {
            std::cout << i << std::endl;
        }
        std::cout << "------" << std::endl;
        std::cout << "CFT" << std::endl;
        for (int t : correctedFixTime) {
            std::cout << t << std::endl;
        }
        std::cout << "------" << std::endl;
    }
    
    int numTimeSteps = 0;
    for (int fTime : correctedFixTime) {
        numTimeSteps += fTime / timeStep;
    }
    if (numTimeSteps < 1) {
        throw std::invalid_argument("Trial response time is smaller than time step");
    }
    numTimeSteps++;

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
    for (float ss = barrierDown.at(0) + (stateStep / 2); ss <= barrierUp.at(0) - (stateStep / 2); ss += stateStep) {
        states.push_back(ss);
    }

    float biasStateVal = MAXFLOAT;
    int biasState = 0;
    for (int i = 0; i < states.size(); i++) {
        float r = abs(states.at(i) - this->bias);
        if (r < biasStateVal) {
            biasState = i;
            biasStateVal = r;
        }
    }

    // Initialize an empty probability state grid
    std::vector<std::vector<double>> prStates; // prStates[state][timeStep]
    for (int i = 0; i < states.size(); i++) {
        prStates.push_back({});
        for (int j = 0; j < numTimeSteps; j++) {
            prStates.at(i).push_back(0);
        }
    }

    // Initialize vectors corresponding to the probability of crossing the 
    // top or bottom barriers at each timestep. 
    std::vector<double> probUpCrossing; 
    std::vector<double> probDownCrossing;
    for (int i = 0; i < numTimeSteps; i++) {
        probUpCrossing.push_back(0);
        probDownCrossing.push_back(0);
    }
    prStates.at(biasState).at(0) = 1; 

    int time = 1;

    // Initialize a change matrix where each value at (i, j) 
    // indicates the difference between states[i] and states[j] 
    std::vector<std::vector<float>> changeMatrix(states.size(), std::vector<float>(states.size())); 
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < states.size(); j++) {
            changeMatrix[i][j] = states[i] - states[j];
        }
    }
    if (debug) {
        printMatrix<float>(changeMatrix, "CHANGE MATRIX");
    }

    // Distance from every state to the top barrier at each timestep
    std::vector<std::vector<float>> changeUp(states.size(), std::vector<float>(numTimeSteps));
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < numTimeSteps; j++) {
            changeUp[i][j] = barrierUp[j] - states[i];
        }
    }
    if (debug) {
        printMatrix<float>(changeUp, "CHANGE UP");
    }


    // Distance from every state to the bottom barrier at each timestep
    std::vector<std::vector<float>> changeDown(states.size(), std::vector<float>(numTimeSteps));
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < numTimeSteps; j++) {
            changeDown[i][j] = barrierDown[j] - states[i];
        }
    }
    if (debug) {
        printMatrix<float>(changeDown, "CHANGE DOWN");
    }

    assert(correctedFixItem.size() == correctedFixTime.size());
    for (int i = 0; i < correctedFixItem.size(); i++) {
        int fItem = correctedFixItem[i];
        int fTime = correctedFixTime[i];

        if (debug) {
            std::cout << "============" << std::endl;
            std::cout << "fItem " << i << ": " << fItem << std::endl;
            std::cout << "fTime " << i << ": " << fTime << std::endl;
            std::cout << "============" << std::endl;
        }

        float mean;
        if (fItem == 1) {
            mean = this->d * (trial.valueLeft - (this->theta * trial.valueRight));
        }
        else if (fItem == 2) {
            mean = this->d * ((this->theta * trial.valueLeft) - trial.valueRight);
        }
        else {
            mean = 0; 
        }

        for (int t = 0; t < fTime / timeStep; t++) {
            // Compute the likelihood of each change in the matrix using a probability density function with parameters mean and sigma. 
            std::vector<std::vector<double>> probDistChangeMatrix(states.size(), std::vector<double>(states.size()));
            for (size_t i = 0; i < states.size(); i++) {
                for (size_t j = 0; j < states.size(); j++) {
                    float x = changeMatrix[i][j];
                    probDistChangeMatrix[i][j] = probabilityDensityFunction(mean, this->sigma, x);
                }
            }
            if (debug) {
                printMatrix<double>(probDistChangeMatrix, "PROBABILITY CHANGE MATRIX");
            }

            // Fetch the probability states for the previous timeStep
            std::vector<double> prTimeSlice(states.size());
            for (size_t i = 0; i < states.size(); i++) {
                prTimeSlice[i] = prStates[i][time - 1];
            }

            // Compute the dot product between the change matrix and previous timeStep's probabilities
            std::vector<double> prStatesNew(states.size());
            for (size_t i = 0; i < states.size(); i++) {
                double row_sum = 0; 
                for (size_t j = 0; j < states.size(); j++) {
                    row_sum += stateStep * probDistChangeMatrix[i][j] * prTimeSlice[j];
                }
                prStatesNew[i] = row_sum;
            }

            // Check for states that are now out-of-bounds based on decay
            for (int i = 0; i < states.size(); i++) {
                if (states[i] > barrierUp[time] || states[i] < barrierDown[time]) {
                    prStatesNew[i] = 0;
                }
            }
            if (debug) {
                std::cout << "PR STATES NEW" << std::endl;
                for (double d : prStatesNew) {
                    std::cout << d << std::endl;
                }
                std::cout << "------" << std::endl;
            }

            std::vector<float> currChangeUp;
            for (auto s : changeUp) {
                currChangeUp.push_back(s.at(time));
            }
            std::vector<double> changeUpCDFs;
            for (int i = 0; i < currChangeUp.size(); i++) {
                float x = currChangeUp[i];
                changeUpCDFs.push_back(
                    1 - cumulativeDensityFunction(mean, this->sigma, x)
                );
            }
            assert(changeUpCDFs.size() == prTimeSlice.size());
            double tempUpCross = 0;
            for (int i = 0; i < prTimeSlice.size(); i++) {
                tempUpCross += changeUpCDFs[i] * prTimeSlice[i];
            }

            std::vector<float> currChangeDown;
            for (auto s: changeDown) {
                currChangeDown.push_back(s.at(time));
            }
            std::vector<double> changeDownCDFs;
            for (int i = 0; i < currChangeDown.size(); i++) {
                float x = currChangeDown[i];
                changeDownCDFs.push_back(
                    cumulativeDensityFunction(mean, this->sigma, x)
                );
            }
            assert(changeDownCDFs.size() == prTimeSlice.size());
            double tempDownCross = 0;
            for (int i = 0; i < prTimeSlice.size(); i++) {
                tempDownCross += changeDownCDFs[i] * prTimeSlice[i];
            }

            double sumIn = 0; 
            for (double prob : prTimeSlice) {
                sumIn += prob; 
            }
            double sumCurrent = tempUpCross + tempDownCross; 
            for (double prob : prStatesNew) {
                sumCurrent += prob;
            }
            double normFactor = sumIn / sumCurrent; 
            for (int i = 0; i < prStatesNew.size(); i++) {
                prStatesNew[i] *= normFactor;
            }
            tempUpCross *= normFactor;
            tempDownCross *= normFactor; 
            for (int i = 0; i < prStates.size(); i++) {
                prStates[i][time] = prStatesNew[i];
            }
            probUpCrossing[time] = tempUpCross; 
            probDownCrossing[time] = tempDownCross;

            time++;
        }
    }
    double likelihood = 0;
    if (trial.choice == -1) {
        if (probUpCrossing[probUpCrossing.size() - 1] > 0) {
            likelihood = probUpCrossing[probUpCrossing.size() - 1];
        }
    }
    else if (trial.choice == 1) {
        if (probDownCrossing[probDownCrossing.size() - 1] > 0) {
            likelihood = probDownCrossing[probDownCrossing.size() - 1];
        }
    }
    return likelihood;
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
    // std::mt19937 gen(SEED);
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
