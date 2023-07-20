#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <string> 
#include <random>
#include <fstream>
#include <iomanip> 
#include "nlohmann/json.hpp"
#include "util.h"
#include "ddm.h"

#ifdef IGNORE_SPACE_CONSTRAINTS
    bool restrictSpace = false; 
#else 
    bool restrictSpace = true; 
#endif 

#ifdef USE_GPU
    bool useGPU = true; 
#else
    bool useGPU = false; 
#endif 

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

double DDM::getTrialLikelihood(DDMTrial trial, bool debug, int timeStep, float approxStateStep) {
    int numTimeSteps = trial.RT / timeStep;
    if (numTimeSteps < 1) {
        throw std::invalid_argument("trial response time is smaller than time step.");
    }
    if (debug) {
        std::cout << std::setprecision(6) << std::fixed;
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
    for (float ss = barrierDown.at(0) + (stateStep / 2); ss <= barrierUp.at(0) - (stateStep / 2); ss += stateStep) {
        states.push_back(ss);
    }

    if (debug) {
        std::cout << "STATES" << std::endl;
        for (float s : states) {
            std::cout << s << " " << std::endl;
        }
        std::cout << "------" << std::endl;
    }
    
    // Get index of state corresponding to the bias
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

    int elapsedNDT = 0;
    bool recomputePDCM = true; 
    float prevMean = 0; 
    std::vector<std::vector<double>> probDistChangeMatrix(states.size(), std::vector<double>(states.size()));

    for (int time = 1; time < numTimeSteps; time++) {
        if (debug) {
            std::cout << "============" << std::endl;
            std::cout << "TIMESTEP " << time << std::endl;
            std::cout << "============" << std::endl;
        }
        float mean;
        if (elapsedNDT < this->nonDecisionTime / timeStep) {
            mean = 0;
            elapsedNDT += 1;
        } else {
            mean = this->d * (trial.valueLeft - trial.valueRight);
        }
        if (debug) {
            std::cout << "mean: " << mean << std::endl;
        }

        if (mean != prevMean) {
            recomputePDCM = true; 
        } else {
            recomputePDCM = false; 
        }
        
        // Compute the likelihood of each change in the matrix using a probability density function with parameters mean and sigma. 
        // Only necessary when: 
        //     -mean of the normal distribution has changed
        //     -first timestep 
        //     -restricting space
        if (recomputePDCM || time == 1 || restrictSpace) {
            for (size_t i = 0; i < states.size(); i++) {
                for (size_t j = 0; j < states.size(); j++) {
                    float x = changeMatrix[i][j];
                    probDistChangeMatrix[i][j] = probabilityDensityFunction(mean, this->sigma, x);
                }
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

        if (debug) {
            std::cout << "PREV TIME SLICE" << std::endl; 
            for (double d : prTimeSlice) {
                std::cout << d << std::endl; 
            }
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
        if (debug) {
            std::cout << "CURR CHANGE UP CDFs" << std::endl; 
            for (float f : changeUpCDFs) {
                std::cout << f << std::endl;
            }
            std::cout << "------" << std::endl;
        }
        assert(changeUpCDFs.size() == prTimeSlice.size());
        double tempUpCross = 0;
        for (int i = 0; i < prTimeSlice.size(); i++) {
            tempUpCross += changeUpCDFs[i] * prTimeSlice[i];
        }
        if (debug) {
            std::cout << "temp up cross: " << tempUpCross << std::endl; 
        }

        std::vector<float> currChangeDown;
        for (auto s: changeDown) {
            currChangeDown.push_back(s.at(time));
        }
        std::vector<double> changeDownCDFs;
        for (int i = 0; i < currChangeDown.size(); i++) {
            float x = currChangeDown[i];
            changeDownCDFs.push_back(
                cumulativeDensityFunction(0, 1, (x - mean) / this->sigma)
            );
        }
        assert(changeDownCDFs.size() == prTimeSlice.size());
        double tempDownCross = 0;
        for (int i = 0; i < prTimeSlice.size(); i++) {
            tempDownCross += changeDownCDFs[i] * prTimeSlice[i];
        }
        if (debug) {
            std::cout << "temp down cross: " << tempDownCross << std::endl; 
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

        if (debug) {
            std::cout << "norm factor " << normFactor << std::endl; 
        }
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

        prevMean = mean; 
    }

    double likelihood = 0; 
    if (trial.choice == -1) {
        if (probUpCrossing[probUpCrossing.size() - 1] > 0) {
            likelihood = probUpCrossing[probUpCrossing.size() - 1];
        } 
    } else if (trial.choice == 1) {
        if (probDownCrossing[probDownCrossing.size() - 1] > 0) {
            likelihood = probDownCrossing[probDownCrossing.size() - 1];
        }
    }    
    assert(likelihood < 1);
    return likelihood;
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

double DDMParallelNLL(DDM ddm, std::vector<DDMTrial> trials) {
    double NLL = 0;
    BS::thread_pool pool;
    BS::multi_future<double> futs = pool.parallelize_loop(
        0, trials.size(), 
        [&ddm, &trials](const int a, const int b) {
            double block_total = 0; 
            for (int i = a; i < b; ++i) {
                block_total += -log(
                    ddm.getTrialLikelihood(trials[i])
                );
            }
            return block_total;
        }
    );
    std::vector<double> totals = futs.get();
    for (const double t : totals) {
        NLL += t; 
    }
    return NLL;
}
