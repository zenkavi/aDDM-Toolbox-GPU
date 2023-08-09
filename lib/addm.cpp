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
#include "ddm.cuh"
#include "util.h"
#include "addm.cuh"


FixationData::FixationData(float probFixLeftFirst, std::vector<int> latencies, 
    std::vector<int> transitions, fixDists fixations) {

    this->probFixLeftFirst = probFixLeftFirst;
    this->latencies = latencies;
    this->transitions = transitions;
    this->fixations = fixations;
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
    unsigned int nonDecisionTime, float bias, float decay) : 
    DDM(d, sigma, barrier, nonDecisionTime, bias, decay) {
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
                fTime -= remainingNDT; 
                correctedFixItem.push_back(fItem);
                correctedFixTime.push_back(max(fTime, 0));
                remainingNDT -= fTime; 
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
    std::vector<float> barrierDown(numTimeSteps);
    if (this->decay != 0) {
        for (int i = 1; i < numTimeSteps; i++) {
            barrierUp.at(i) = this->barrier / (1 + (this->decay * i));
            barrierDown.at(i) = -this->barrier / (1 + (this->decay * i));
        }
    } else {
        std::fill(barrierUp.begin(), barrierUp.end(), this->barrier);
        std::fill(barrierDown.begin(), barrierDown.end(), -this->barrier);
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
        pmat<float>(changeMatrix, "CHANGE MATRIX");
    }

    // Distance from every state to the top barrier at each timestep
    std::vector<std::vector<float>> changeUp(states.size(), std::vector<float>(numTimeSteps));
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < numTimeSteps; j++) {
            changeUp[i][j] = barrierUp[j] - states[i];
        }
    }
    if (debug) {
        pmat<float>(changeUp, "CHANGE UP");
    }


    // Distance from every state to the bottom barrier at each timestep
    std::vector<std::vector<float>> changeDown(states.size(), std::vector<float>(numTimeSteps));
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < numTimeSteps; j++) {
            changeDown[i][j] = barrierDown[j] - states[i];
        }
    }
    if (debug) {
        pmat<float>(changeDown, "CHANGE DOWN");
    }

    assert(correctedFixItem.size() == correctedFixTime.size());
    std::vector<std::vector<double>> probDistChangeMatrix(states.size(), std::vector<double>(states.size()));
    for (int c = 0; c < correctedFixItem.size(); c++) {
        int fItem = correctedFixItem[c];
        int fTime = correctedFixTime[c];

        if (debug) {
            std::cout << "============" << std::endl;
            std::cout << "fItem " << c << ": " << fItem << std::endl;
            std::cout << "fTime " << c << ": " << fTime << std::endl;
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

        for (size_t i = 0; i < states.size(); i++) {
            for (size_t j = 0; j < states.size(); j++) {
                float x = changeMatrix[i][j];
                probDistChangeMatrix[i][j] = probabilityDensityFunction(mean, this->sigma, x);
            }
        }
        if (debug) {
            pmat<double>(probDistChangeMatrix, "PROBABILITY CHANGE MATRIX");
        }
        for (int t = 0; t < fTime / timeStep; t++) {
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
    if (likelihood == 0) {
        likelihood = pow(10, -20);
    }
    return likelihood;
}


aDDMTrial aDDM::simulateTrial(
    int valueLeft, int valueRight, FixationData fixationData, int timeStep, 
    int numFixDists, fixDists fixationDist, vector<int> timeBins, int seed) {

    std::vector<int> fixItem;
    std::vector<int> fixTime;
    std::vector<float> fixRDV;

    std::random_device rd;
    std::mt19937 gen(seed == -1 ? rd() : seed); 

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
            if (fixationDist.empty()) {
                vector<float> fixTimes = fixationData.fixations.at(fixNumber);
                std::uniform_int_distribution<std::size_t> fudist(0, fixTimes.size() - 1);
                rIDX = fudist(gen);
                currFixTime = fixTimes.at(rIDX);
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


ProbabilityData aDDM::computeParallelNLL(std::vector<aDDMTrial> trials, bool debug, int timeStep, float approxStateStep) {
    ProbabilityData datasetTotals = ProbabilityData(0, 0);
    BS::thread_pool pool;
    std::vector<double> trialLikelihoods(trials.size());
    BS::multi_future<ProbabilityData> futs = pool.parallelize_loop(
        0, trials.size(), 
        [this, &trials, debug, timeStep, approxStateStep, &trialLikelihoods](const int a, const int b) {
            ProbabilityData aux = ProbabilityData(0, 0);
            for (int i = a; i < b; ++i) {
                double prob = this->getTrialLikelihood(trials[i], debug, timeStep, approxStateStep);
                trialLikelihoods[i] = prob; 
                aux.likelihood += prob; 
                aux.NLL += -log(prob);
            }
            return aux;
        }
    );
    std::vector<ProbabilityData> totals = futs.get();
    for (const ProbabilityData t : totals) {
        datasetTotals.NLL += t.NLL;
        datasetTotals.likelihood += t.likelihood; 
    }
    datasetTotals.trialLikelihoods = trialLikelihoods; 
    return datasetTotals;
}


void aDDMTrial::writeTrialsToCSV(std::vector<aDDMTrial> trials, string filename) {
    std::ofstream fp;
    fp.open(filename);
    fp << "trial,choice,rt,valueLeft,valueRight,fixItem,fixTime\n";
    int id = 0; 

    for (aDDMTrial adt : trials) {
        assert(adt.fixItem.size() == adt.fixTime.size());
        for (int i = 0; i < adt.fixItem.size(); i++) {
            fp << id << "," << adt.choice << "," << adt.RT << "," << 
                adt.valueLeft << "," << adt.valueRight << "," <<
                adt.fixItem[i] << "," << adt.fixTime[i] << "\n";
        }
        id++;
    }
    fp.close();    
}


vector<aDDMTrial> aDDMTrial::loadTrialsFromCSV(string filename) {
    std::vector<aDDMTrial> trials; 
    std::vector<aDDM> addms;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);

    int ID;
    int choice; 
    int RT; 
    int valueLeft;
    int valueRight;
    int prevID;
    int fItem;
    int fTime; 
    bool firstIter = true; 
    std::vector<int> fixItem;
    std::vector<int> fixTime; 

    aDDMTrial adt = aDDMTrial();
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::getline(ss, field, ',');
        ID = std::stoi(field);
        std::getline(ss, field, ',');
        choice = std::stoi(field);
        std::getline(ss, field, ',');
        RT = std::stoi(field);
        std::getline(ss, field, ',');
        valueLeft = std::stoi(field);
        std::getline(ss, field, ',');
        valueRight = std::stoi(field);
        std::getline(ss, field, ',');
        fItem = std::stoi(field);
        std::getline(ss, field, ',');
        fTime = std::stoi(field);
        if (ID == prevID && !firstIter) {
            adt.fixItem.push_back(fItem);
            adt.fixTime.push_back(fTime);
        } else {
            if (firstIter) {
                firstIter = false; 
            } else {
                trials.push_back(adt);
            }
            adt = aDDMTrial(RT, choice, valueLeft, valueRight);
            adt.fixItem.push_back(fItem);
            adt.fixTime.push_back(fTime);

        }
        prevID = ID;
    }
    trials.push_back(adt);
    file.close();
    return trials;
}


MLEinfo<aDDM> aDDM::fitModelMLE(
    std::vector<aDDMTrial> trials, 
    std::vector<float> rangeD, 
    std::vector<float> rangeSigma, 
    std::vector<float> rangeTheta, 
    float barrier, 
    std::string computeMethod, 
    bool normalizePosteriors) {

    if (std::find(validComputeMethods.begin(), validComputeMethods.end(), computeMethod) == validComputeMethods.end()) {
        throw std::invalid_argument("Input computeMethod is invalid.");
    }

    sort(rangeD.begin(), rangeD.end());
    sort(rangeSigma.begin(), rangeSigma.end());
    sort(rangeTheta.begin(), rangeTheta.end()); 

    std::vector<aDDM> potentialModels; 
    for (float d : rangeD) {
        for (float sigma : rangeSigma) {
            for (float theta : rangeTheta) {
                aDDM addm = aDDM(d, sigma, theta, barrier);
                potentialModels.push_back(addm);
            }
        }
    }

    std::function<ProbabilityData(aDDM)> NLLcomputer; 
    if (computeMethod == "basic") {
        NLLcomputer = [trials](aDDM addm) -> ProbabilityData {
            ProbabilityData data = ProbabilityData(); 
            for (aDDMTrial trial : trials) {
                double prob = addm.getTrialLikelihood(trial); 
                data.likelihood += prob; 
                data.trialLikelihoods.push_back(prob);
                data.NLL += -log(prob);
            }
            return data; 
        };
    }
    else if (computeMethod == "thread") {
        NLLcomputer = [trials](aDDM addm) -> ProbabilityData {
            return addm.computeParallelNLL(trials);
        };
    }
    else if (computeMethod == "gpu") {
        if (gpuInvalid) {
            throw std::invalid_argument("CUDA calls disabled.");
        }
#ifndef EXCLUDE_CUDA_CODE
        NLLcomputer = [trials](aDDM addm) -> ProbabilityData {
            return addm.computeGPUNLL(trials);
        };
#endif 
    }

    double minNLL = __DBL_MAX__; 
    std::map<aDDM, ProbabilityData> allTrialLikelihoods; 
    std::map<aDDM, float> posteriors; 
    double numModels = rangeD.size() * rangeSigma.size() * rangeTheta.size();
    std::cout << "num models " << numModels << std::endl;  

    aDDM optimal = aDDM(); 
    for (aDDM addm : potentialModels) {
        ProbabilityData aux = NLLcomputer(addm);
        if (normalizePosteriors) {
            allTrialLikelihoods.insert({addm, aux});
            posteriors.insert({addm, 1 / numModels});
        } else {
            posteriors.insert({addm, aux.NLL});
        }

        std::cout << "testing d=" << addm.d << " sigma=" << addm.sigma << " theta=" << addm.theta << " NLL=" << aux.NLL << std::endl; 
        if (aux.NLL < minNLL) {
            minNLL = aux.NLL; 
            optimal = addm; 
        }
    }
    if (normalizePosteriors) {
        for (int tn = 0; tn < trials.size(); tn++) {
            double denominator = 0; 
            for (const auto &addmPD : allTrialLikelihoods) {
                aDDM curr = addmPD.first; 
                ProbabilityData data = addmPD.second; 
                double likelihood = data.trialLikelihoods[tn];
                denominator += posteriors[curr] * likelihood; 
            }
            std::cout << "denominator " << denominator << std::endl; 
            double sum = 0; 
            for (const auto &addmPD : allTrialLikelihoods) {
                aDDM curr = addmPD.first; 
                ProbabilityData data = addmPD.second; 
                double prior = posteriors[curr];
                double newLikelihoood = data.trialLikelihoods[tn] * prior / denominator; 
                posteriors[curr] = newLikelihoood; 
                sum += newLikelihoood;
            }
            if (sum != 1) {
                double normalizer = 1 / sum; 
                std::cout << "normalizing with normalizer=" << normalizer << std::endl; 
                for (auto &p : posteriors) {
                    p.second *= normalizer; 
                }
            }
        }

        for (const auto &pair : posteriors) {
            aDDM curr = pair.first; 
            double l = pair.second; 
            std::cout << curr.d << " " << curr.sigma << " " << curr.theta << " " << l << std::endl; 
        }
    }
    MLEinfo<aDDM> info;
    info.optimal = optimal; 
    info.likelihoods = posteriors; 
    return info;   
}

