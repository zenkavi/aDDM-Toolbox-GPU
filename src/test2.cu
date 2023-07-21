#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdio.h>
#include "../include/ddm.h"
#include "../include/util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



float d = 0.005;
float sigma = 0.07;
int barrier = 1;
int valueLeft = 3;

int TS = 10; 
float ASS = 0.1; 
float DEC = 0; 
int NDT = 0; 

__device__ int __RC2IDX(int row, int col, int columns_per_row) {
    return (row * columns_per_row) + col; 
}

__device__ double pdf(float x, float mean, float sigma) {
    double first = exp(-0.5 * pow((x - mean) / sigma, 2));
    double second = sigma * sqrt(2 * M_PI);
    return first / second; 
}

// CUDA Kernel to calculate trial likelihoods in parallel
__global__
void getTrialLikelihoodKernel(bool debug, int *RTs, int *choices, int *valDiffs, double* likelihoods, int numTrials, float d, float sigma, int barrier, int nonDecisionTime, int timeStep, float approxStateStep, float dec) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("TID = %i\n", tid);

    if (tid < numTrials) {

    // for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numTrials; tid += blockDim.x * gridDim.y) {

        // printf("entering for tid %i\n", tid);
        int choice = choices[tid];
        int RT = RTs[tid];
        int valDiff = valDiffs[tid];


        // printf("choice %i, RT %i, vd %i\n", choice, RT, valDiff);

        float bias = 0; 

        int numTimeSteps = RT / timeStep; 

        // requires compute capability 2.x
        float* barrierUp = new float[numTimeSteps];
        float *barrierDown = new float[numTimeSteps];

        for (int i = 0 ; i < numTimeSteps; i++) {
            barrierUp[i] = barrier / (1 + (dec * i));
            barrierDown[i] = -barrier / (1 + (dec * i));
        }

        int halfNumStateBins = ceil(barrier / approxStateStep); 
        if (debug) printf("half num state bins %i\n", halfNumStateBins);
        float stateStep = barrier / (halfNumStateBins + 0.5);
        if (debug) printf("state step %f\n", stateStep);
        int numStates = 2 * halfNumStateBins + 1; 


        float *states = new float[numStates];
        int s = 0; 
        float biasStateVal = MAXFLOAT; 
        int biasState; 
        float r; 
        for (float ss = barrierDown[0] + (stateStep / 2); ss <= barrierUp[0] - (stateStep / 2); ss += stateStep) {
            states[s] = ss;
            r = abs(ss - bias); 
            if (r < biasStateVal) {
                biasState = s;
                biasStateVal = r; 
            }
            s++;
        }

        if (debug) {
            for (int i = 0; i < numStates; i++) {
                printf("states[%i] = %f\n", i, states[i]);
            }
            printf("bias state %i\n", biasState);
        }

        double *prStates = new double[numStates];
        for (int i = 0; i < numStates; i++) {
            prStates[i] = (i == biasState) ? 1 : 0; 
        }

        double *probUpCrossing = new double[numTimeSteps];
        double *probDownCrossing = new double[numTimeSteps];
        for (int i = 0; i < numTimeSteps; i++) {
            probUpCrossing[i] = 0; 
            probDownCrossing[i] = 0; 
        }


        if (debug) {
            for (int i = 0 ; i < numStates ; i++) {
                printf("prStates[%i] = %f\n", i, prStates[i]);
            }
        }
        

        float *changeMatrix = new float[numStates * numStates];
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numStates; j++) {
                changeMatrix[__RC2IDX(i, j, numStates)] = states[i] - states[j];
            }
        }

        float *changeUp = new float[numStates * numTimeSteps];
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numTimeSteps; j++) {
                changeUp[__RC2IDX(i, j, numTimeSteps)] = barrierUp[j] - states[i];
            }
        }

        float *changeDown = new float[numStates * numTimeSteps];
        for (int i = 0; i < numStates; i++) {
            for (int j = 0; j < numTimeSteps; j++) {
                changeDown[__RC2IDX(i, j, numTimeSteps)] = barrierDown[j] - states[i];
            }
        }

        if (debug) {
            printf("change matrix\n");
            for (int i = 0; i < numStates * numStates; i++) {
                printf("%f ", changeMatrix[i]);
                if ((i + 1) % numStates == 0) {
                    printf("\n");
                }
            }

            printf("change up\n");
            for (int i = 0; i < numStates * numTimeSteps; i++) {
                printf("%f ", changeUp[i]);
                if ((i + 1) % numTimeSteps == 0) {
                    printf("\n");
                }
            }
        }


        int elapsedNDT = 0;
        bool recomputePDCM = true; 
        float prevMean = 0; 
        float *probDistChangeMatrix = new float[numStates * numStates];

        for (int time = 1; time < numTimeSteps; time++) {

            if (debug) printf(
                "============\n timestep %i \n============", time
            );

            float mean; 
            if (elapsedNDT < nonDecisionTime / timeStep) {
                mean = 0; 
                elapsedNDT += 1; 
            } else {
                mean = d * valDiff;
            }

            if (mean != prevMean) {
                recomputePDCM = true;
            } else {
                recomputePDCM = false; 
            }

            if (recomputePDCM || time == 1) {
                for (int i = 0; i < numStates; i++) {
                    for (int j = 0; j < numStates; j++) {
                        float x = changeMatrix[__RC2IDX(i, j, numStates)];
                        probDistChangeMatrix[__RC2IDX(i, j, numStates)] = pdf(x, mean, sigma);
                    }
                }
            }

            if (debug) {
                printf("PDCM\n");
                for (int i = 0; i < numStates * numStates; i++) {
                    printf("%f ", probDistChangeMatrix[i]);
                    if ((i + 1) % numStates == 0) {
                        printf("\n");
                    }
                }
            }

            double rowSum; 
            double* prStatesNew = new double[numStates];
            for (int i = 0; i < numStates; i++) {
                rowSum = 0; 
                for (int j = 0; j < numStates; j++) {
                    rowSum += stateStep * probDistChangeMatrix[__RC2IDX(i, j, numStates)] * prStates[j];
                }
                prStatesNew[i] = (states[i] > barrierUp[time] || states[i] < barrierDown[time]) ? 0 : rowSum;
            }

            if (debug) {
                for (int i = 0 ; i < numStates ; i++) {
                    printf("prStatesNew[%i] = %f\n", i, prStatesNew[i]);
                }
            }

            float *changeUpCDFs = new float[numStates];
            for (int i = 0; i < numStates; i++) {
                float x = changeUp[__RC2IDX(i, time, numTimeSteps)];
                changeUpCDFs[i] = 1 - normcdf((x - mean) / sigma);
            }
            if (debug) {
                for (int i = 0; i < numStates; i++) {
                    printf("changeUpCDFs[%i] = %f\n", i, changeUpCDFs[i]);
                }
            }
            double tempUpCross = 0; 
            for (int i = 0; i < numStates; i++) {
                tempUpCross += changeUpCDFs[i] * prStates[i];
            }

            float *changeDownCDFs = new float[numStates];
            for (int i = 0; i < numStates; i++) {
                float x = changeDown[__RC2IDX(i, time, numTimeSteps)];
                changeDownCDFs[i] = normcdf((x - mean) / sigma);
            }
            if (debug) {
                for (int i = 0; i < numStates; i++) {
                    printf("changeDownCDFs[%i] = %f\n", i, changeDownCDFs[i]);
                }
            }
            double tempDownCross = 0; 
            for (int i = 0; i < numStates; i++) {
                tempDownCross += changeDownCDFs[i] * prStates[i];
            }

            if (debug) printf("temp up cross = %f\n", tempUpCross);
            if (debug) printf("temp down cross = %f\n", tempDownCross);

            double sumIn = 0; 
            double sumCurrent = tempUpCross + tempDownCross; 
            for (int i = 0; i < numStates; i++) {
                sumIn += prStates[i];
                sumCurrent += prStatesNew[i];
            }
            double normFactor = sumIn / sumCurrent; 
            for (int i = 0; i < numStates; i++) {
                prStates[i] = prStatesNew[i] * normFactor; 
            }

            probUpCrossing[time] = tempUpCross * normFactor; 
            probDownCrossing[time] = tempDownCross * normFactor;

            prevMean = mean;

            // delete prStatesNew;
            // delete changeUpCDFs;
            // delete changeDownCDFs;
        }

        double likelihood = 0; 
        if (choice == -1) {
            if (probUpCrossing[numTimeSteps - 1] > 0) {
                likelihood = probUpCrossing[numTimeSteps - 1];
            }
        } else if (choice == 1) {
            if (probDownCrossing[numTimeSteps - 1] > 0) {
                likelihood = probDownCrossing[numTimeSteps - 1];
            }
        }

        // delete barrierUp;
        // delete barrierDown;
        // delete probUpCrossing;
        // delete probDownCrossing;
        // delete states;
        // delete changeMatrix;
        // delete changeUp;
        // delete changeDown;
        // delete probDistChangeMatrix;
        
        printf("computed likelihood: %f\n", likelihood);
        likelihoods[tid] = likelihood;

    }
    
}

void callGetTrialLikelihoodKernel(bool debug, int numBlocks, int numThreadsPerBlock, DDMTrial *trials, double *likelihoods, int numTrials, float d, float sigma, float barrier, int nonDecisionTime, int timeStep, float approxStateStep, float dec) {
    int *d_RTs, *d_choices, *d_VDs;
    cudaMalloc((void**)&d_RTs, numTrials * sizeof(int));
    cudaMalloc((void**)&d_choices, numTrials * sizeof(int));
    cudaMalloc((void**)&d_VDs, numTrials * sizeof(int));

    int *h_VDs = new int[numTrials];
    int *h_RTs = new int[numTrials];
    int *h_choices = new int[numTrials];
    for (int i = 0; i < numTrials; i++) {
        h_VDs[i] = trials[i].valueLeft - trials[i].valueRight;
        h_RTs[i] = trials[i].RT;
        h_choices[i] = trials[i].choice;
    }

    cudaMemcpy(d_RTs, &(h_RTs[0]), numTrials * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_choices, &(h_choices[0]), numTrials * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_VDs, &(h_VDs[0]), numTrials * sizeof(int), cudaMemcpyHostToDevice);

    getTrialLikelihoodKernel<<<numBlocks, numThreadsPerBlock>>>(
        debug,
        d_RTs,
        d_choices,
        d_VDs,
        likelihoods,
        numTrials,
        d, sigma, barrier,
        nonDecisionTime,
        timeStep,
        approxStateStep,
        dec
    );

    cudaFree(d_RTs);
    cudaFree(d_choices);
    cudaFree(d_VDs);
    // delete[] h_RTs;
    // delete[] h_choices;
    // delete[] h_VDs;
}


int main() {
    std::vector<DDMTrial> hostTrials; 
    std::ifstream file("results/test.csv");
    std::string line;
    std::getline(file, line);
    int choice; 
    int RT; 
    int valDiff;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::getline(ss, field, ',');
        choice = std::stoi(field);
        std::getline(ss, field, ',');
        RT = std::stoi(field);
        std::getline(ss, field, ',');
        valDiff = std::stoi(field);
        DDMTrial dt = DDMTrial(RT, choice, valueLeft, valueLeft - valDiff);
        hostTrials.push_back(dt);
    }
    file.close();
    std::cout << "Counted " << hostTrials.size() << " trials." << std::endl;

    // Move data to GPU memory
    int numTrials = hostTrials.size();
    DDMTrial* d_trials;
    double* d_likelihoods;
    cudaMalloc((void**)&d_trials, numTrials * sizeof(DDMTrial));
    cudaMalloc((void**)&d_likelihoods, numTrials * sizeof(double));
    cudaMemcpy(d_trials, hostTrials.data(), numTrials * sizeof(DDMTrial), cudaMemcpyHostToDevice);

    int numThreadsPerBlock = 256;
    int numBlocks = (numTrials + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Launch the CUDA kernel
    callGetTrialLikelihoodKernel(false, numBlocks, numThreadsPerBlock, hostTrials.data(), d_likelihoods, hostTrials.size(), d, sigma, barrier, NDT, TS, ASS, DEC);

    // Copy results back to the CPU
    std::vector<double> h_likelihoods(numTrials);
    cudaMemcpy(h_likelihoods.data(), d_likelihoods, numTrials * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numTrials; i++) {
        std::cout << "likelihood " << h_likelihoods[i] << std::endl;
    }



    for (int i = 0; i < numTrials; ++i) {
        hostTrials[i].likelihood = h_likelihoods[i];
    }

    // Free allocated device memory
    cudaFree(d_trials);
    cudaFree(d_likelihoods);

    // Write results to file
    std::ofstream fp;
    fp.open("results/ddm_gpu_simulations_prob.csv");
    fp << "choice,RT,p\n";
    double NLL = 0;
    for (DDMTrial dt : hostTrials) {
        fp << dt.choice << "," << dt.RT << "," << dt.likelihood << "\n";
        NLL += -log(dt.likelihood);
    }
    fp.close();

    std::cout << "NLL: " << NLL << std::endl;

    return 0;
}