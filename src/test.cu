#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <cublas_v2.h>
#include <cuda.h>
#include "../include/ddm.h"
#include "../include/util.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

std::vector<float> rangeD = {0.003};
std::vector<float> rangeSigma = {0.03};

int barrier = 1;
int valueLeft = 3; 

int timeStep = 10;
float approxStateStep = 0.1; 

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<DDMTrial> trials;
    std::vector<DDM> ddms;
    std::ifstream file("results/ddm_simulations.csv");
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
        trials.push_back(dt);
    }
    file.close();
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    for (float d : rangeD) {
        for (float sigma : rangeSigma) {
            ddms.push_back(DDM(d, sigma, barrier));
        }
    }


    for (DDM& ddm : ddms) {
        double NLL = 0; 
        for (DDMTrial& trial : trials) {
            double prob = 0; 


            int numTimeSteps = trial.RT / timeStep;
            std::cout << numTimeSteps << std::endl; 


            // // float *hostBarrierUp = new float[numTimeSteps];
            // // float *hostBarrierDown = new float[numTimeSteps];

            std::vector<float> hostBarrierUp(numTimeSteps);
            std::vector<float> hostBarrierDown(numTimeSteps);

            std::cout << "HERE" << std::endl;
            for (int i = 0; i < numTimeSteps; i++) {
                hostBarrierUp[i] = ddm.barrier / (1 + (DECAY * i));
                hostBarrierDown[i] = -ddm.barrier / (1 + (DECAY * i));
            }

            float *devBarrierUp;
            float *devBarrierDown; 
            cudaMalloc((void**) &devBarrierUp, numTimeSteps * sizeof(float));
            cudaMalloc((void**) &devBarrierDown, numTimeSteps * sizeof(float));
            cublasSetVector(numTimeSteps, sizeof(float), &hostBarrierUp[0], 1, devBarrierUp, 1);
            cublasSetVector(numTimeSteps, sizeof(float), &hostBarrierDown[0], 1, devBarrierDown, 1);

            int halfNumStateBins = ceil(ddm.barrier / approxStateStep);
            float stateStep = ddm.barrier / (halfNumStateBins + 0.5);

            // // float *states = new float[numTimeSteps];

            std::vector<float> hostStates(numTimeSteps);
            float biasStateVal = MAXFLOAT; 
            int biasState = 0; 
            int i = 0;
            for (float ss = hostBarrierDown[0] + (stateStep / 2); ss <= hostBarrierUp[0] - (stateStep / 2); ss += stateStep) {
                hostStates.push_back(ss);
                if (abs(ss - ddm.bias) < biasStateVal) {
                    biasState = i;
                    biasStateVal = abs(ss - ddm.bias);
                }
                i++;
            }

            float *devStates; 
            int numStates = hostStates.size();
            cudaMalloc((void **) &devStates, numStates * sizeof(float));
            cublasSetVector(numStates, sizeof(float), &hostStates[0], 1, devStates, 1);

            std::vector<float> hostChangeMatrix(numStates * numStates, 0.0f);
            for (size_t i = 0; i < numStates; i++) {
                for (size_t j = 0; j < numStates; j++) {
                    hostChangeMatrix[i * numStates + j] = hostStates[i] = hostStates[j];
                }
            }
            float *devChangeMatrix; 
            cudaMalloc((void **) &devChangeMatrix, numStates * numStates * sizeof(float));
            if (cublasSetMatrix(numStates, numStates, sizeof(float), &hostChangeMatrix[0], numStates, devChangeMatrix, numStates) != CUBLAS_STATUS_SUCCESS) {
                std::cout << "Set Change Matrix failed" << std::endl;
                cudaFree(devBarrierUp);
                cudaFree(devBarrierDown);
                cudaFree(devStates);
                cudaFree(devChangeMatrix);
                exit(1);
            }



            cudaFree(devBarrierUp);
            cudaFree(devBarrierDown);
            cudaFree(devStates);
            cudaFree(devChangeMatrix);
        }
    }

    return 0;
}