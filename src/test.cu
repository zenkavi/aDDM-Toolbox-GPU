#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include "../include/ddm.h"
#include "../include/util.h"


std::vector<float> rangeD = {0.005};
std::vector<float> rangeSigma = {0.07};

int barrier = 1;
int valueLeft = 3; 

int timeStep = 10;
float approxStateStep = 0.1; 


struct abs_diff_functor
{
    const float bias;

    __host__ __device__
    float operator()(const float& x) const
    {
        return fabs(x - bias);
    }
};

struct cdf_functor
{
    float a;
    float b; 
    float m; 
    float s;

    cdf_functor(float _a, float _b, float _m, float _s) : a(_a), b(_b), m(_m), s(_s) {}

    __host__ __device__ 
    float operator()(const float& x) const 
    {
        return a + b * normcdf((x - m) / s);
    }
};

__device__ double devicePDF(float x, float mean, float sigma) {
    float first = exp(-0.5 * pow((x - mean) / sigma, 2));
    float second = sigma * sqrt(2 * M_PI);
    return first / second;
}

__global__ void computeChangeMatrix(const float* states, float* changeMatrix, size_t numStates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numStates && j < numStates) {
        int index = i * numStates + j;
        changeMatrix[index] = states[j] - states[i];
    }
}

__global__ void computeChange(const float* barrier, const float* states, float* changeMat, size_t numStates, size_t numTimeSteps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numStates && j < numTimeSteps) {
        int index = j * numStates + i;
        changeMat[index] = barrier[j] - states[i];
    }
}

__global__ void computeProbDistChangeMatrix(const float* changeMatrix, double *probDistChangeMatrix, size_t numStates, float mean, float sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numStates && j < numStates) {
        int index = j * numStates + i;
        float x = changeMatrix[index];
        double y = devicePDF(x, mean, sigma);
        probDistChangeMatrix[index] = y;
    }    
}

__global__ void computePrStatesNew(const double* probDistChangeMatrix, const double* prevTimeSlice, double* prStatesNew, size_t numStates, float stateStep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numStates) {
        double dotProduct = 0.0;
        for (int j = 0; j < numStates; j++) {
            int index = j * numStates + i;
            dotProduct += probDistChangeMatrix[index] * prevTimeSlice[j];
        }
        prStatesNew[i] = dotProduct * stateStep;
    }
}




double getTrialLikelihoodGPU(DDM ddm, DDMTrial trial) {

    int timeStep = 10;
    float approxStateStep = 0.1;

    int numTimeSteps = trial.RT / timeStep;

    std::cout << std::setprecision(6) << std::fixed; 

    // std::cout << "RT = " << trial.RT << std::endl; 

    thrust::device_vector<float> barrierUp(numTimeSteps, ddm.barrier);
    thrust::device_vector<float> barrierDown(numTimeSteps, -ddm.barrier);

    int halfNumStateBins = ceil(ddm.barrier / approxStateStep);
    float stateStep = ddm.barrier / (halfNumStateBins + 0.5);

    thrust::device_vector<float> states(1 + (barrierUp[0] - barrierDown[0] - (stateStep / 2)) / stateStep);
    thrust::sequence(states.begin(), states.end(), barrierDown[0] + stateStep / 2, stateStep);

    // for (int i = 0; i < states.size(); i++) {
    //     std::cout << "B[" << i << "] = " << states[i] << std::endl;
    // }

    int numStates = states.size();
    thrust::device_vector<float> differences(numStates);
    thrust::transform(states.begin(), states.end(), differences.begin(), thrust::placeholders::_1 - ddm.bias);
    thrust::transform(differences.begin(), differences.end(), differences.begin(), abs_diff_functor{ddm.bias});
    auto min_diff_iter = thrust::min_element(differences.begin(), differences.end());
    int bias_idx = min_diff_iter - differences.begin();

    dim3 blockSize(16, 16);  // Adjust the block size according to your GPU architecture
    dim3 gridSize((numStates + blockSize.x - 1) / blockSize.x, (numStates + blockSize.y - 1) / blockSize.y);

    // std::cout << "bias index " << bias_idx << std::endl;

    // Column Major Ordering 
    // prStates[state][time] = prStates[time * numStates + state]
    // flattened matrix: all timesteps for a given state sequentially
    thrust::device_vector<double> prStates(numStates * numTimeSteps, 0); 
    prStates[bias_idx] = 1;

    thrust::device_vector<double> probUpCrossing(numTimeSteps, 0);
    thrust::device_vector<double> probDownCrossing(numTimeSteps, 0);

    // changeMatrix[i][j] = changeMatrix[i * numStates + j]
    thrust::device_vector<float> changeMatrix(numStates * numStates);

    thrust::device_vector<float> changeUp(numStates * numTimeSteps);    
    thrust::device_vector<float> changeDown(numStates * numTimeSteps);
    for (size_t i = 0; i < numStates; i++) {
        for (size_t j = 0; j < numTimeSteps; j++) {
            changeUp[j * numStates + i] = barrierUp[j] - states[i];
        }
    }
    for (size_t i = 0; i < numStates; i++) {
        for (size_t j = 0; j < numTimeSteps; j++) {
            changeDown[j * numStates + i] = barrierDown[j] - states[i];
        }
    }

    computeChangeMatrix<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(states.data()), 
        thrust::raw_pointer_cast(changeMatrix.data()), 
        numStates);
    computeChange<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(barrierUp.data()), 
        thrust::raw_pointer_cast(states.data()), 
        thrust::raw_pointer_cast(changeUp.data()), 
        numStates, numTimeSteps);
    computeChange<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(barrierDown.data()), 
        thrust::raw_pointer_cast(states.data()), 
        thrust::raw_pointer_cast(changeDown.data()), 
        numStates, numTimeSteps);

    std::cout << "CHANGE MATRIX" << std::endl; 
    for (int i = 0; i < changeMatrix.size(); i++) {
        std::cout << changeMatrix[i] << " "; 
        if ((i + 1) % numStates == 0) { std::cout << std::endl; }
    }
    std::cout << "CHANGE UP" << std::endl; 
    for (int i = 0; i < changeUp.size(); i++) {
        std::cout << changeUp[i] << " "; 
        if ((i + 1) % numStates == 0) { std::cout << std::endl; }
    }
    std::cout << "CHANGE DOWN" << std::endl; 
    for (int i = 0; i < changeDown.size(); i++) {
        std::cout << changeDown[i] << " "; 
        if ((i + 1) % numStates == 0) { std::cout << std::endl; }
    }

    int elapsedNDT = 0; 
    for (int time = 1; time < numTimeSteps; time++) {
        std::cout << "============" << std::endl;
        std::cout << "TIMESTEP " << time << std::endl;
        std::cout << "============" << std::endl;        


        float mean; 
        if (elapsedNDT < ddm.nonDecisionTime / timeStep) {
            mean = 0; 
            elapsedNDT += 1; 
        } else {
            mean = ddm.d * (trial.valueLeft - trial.valueRight);
        }

        thrust::device_vector<double> probDistChangeMatrix(numStates * numStates);
        computeProbDistChangeMatrix<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(changeMatrix.data()), 
            thrust::raw_pointer_cast(probDistChangeMatrix.data()), 
            numStates, mean, ddm.sigma
        );

        std::cout << "PROB DIST CHANGE" << std::endl; 
        for (int i = 0; i < probDistChangeMatrix.size(); i++) {
            std::cout << probDistChangeMatrix[i] << " "; 
            if ((i + 1) % numStates == 0) { std::cout << std::endl; }
        }

        thrust::device_vector<double> prevTimeSlice(numStates);
        thrust::copy(
            prStates.begin() + (time - 1) * numStates, 
            prStates.begin() + (time) * numStates, 
            prevTimeSlice.begin()
        );

        std::cout << "PREV TIME SLICE" << std::endl; 
        for (double f : prevTimeSlice) { 
            std::cout << f << std::endl; 
        }
        std::cout << "numStates " << numStates << std::endl; ;
        std::cout << "time slice " << prevTimeSlice.size() << std::endl; 

        thrust::device_vector<double> prStatesNew(numStates);

        computePrStatesNew<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(probDistChangeMatrix.data()), 
            thrust::raw_pointer_cast(prevTimeSlice.data()), 
            thrust::raw_pointer_cast(prStatesNew.data()), 
            numStates, stateStep
        );
        std::cout << "PR STATES NEW" << std::endl; 
        for (double d : prStatesNew) {
            std::cout << d << std::endl; 
        }

        thrust::device_vector<float> currChangeUp(numStates); 
        thrust::transform(
            changeUp.begin() + (time) * numTimeSteps, 
            changeUp.begin() + (time + 1) * numTimeSteps, 
            currChangeUp.begin(), 
            cdf_functor(1, -1, mean, ddm.sigma)
        );
        std::cout << "CURR CHANGE UP (CDF)" << std::endl; 
        for (float f : currChangeUp) {
            std::cout << f << std::endl; 
        }

        double tempUpCross = 0; 
        for (int i = 0; i < numStates; i++) {
            tempUpCross += currChangeUp[i] * prevTimeSlice[i];
        }
        // double tempUpCross = thrust::inner_product(currChangeUp.begin(), currChangeUp.end(), prevTimeSlice.begin(), 0);
        std::cout << "temp up cross " << tempUpCross << std::endl; 

        thrust::device_vector<float> currChangeDown(numStates);
        thrust::transform(
            changeDown.begin() + (time) * numTimeSteps, 
            changeDown.begin() + (time + 1) * numTimeSteps, 
            currChangeDown.begin(),
            cdf_functor(0, 1, mean, ddm.sigma)
        );
        std::cout << "CURR CHANGE DOWN (CDF)" << std::endl; 
        for (float f : currChangeDown) { 
            std::cout << f << std::endl; 
        }

        double tempDownCross = 0; 
        for (int i = 0; i < numStates; i++) {
            tempDownCross += currChangeDown[i] * prevTimeSlice[i];
        }
        // double tempDownCross = thrust::inner_product(currChangeDown.begin(), currChangeDown.end(), prevTimeSlice.begin(), 0);
        std::cout << "temp down cross " << tempDownCross << std::endl; 

        double sumIn = thrust::reduce(prevTimeSlice.begin(), prevTimeSlice.end(), 0, thrust::plus<double>());
        double sumCurrent = thrust::reduce(
            prStatesNew.begin(), prStatesNew.end(), 0, thrust::plus<double>()
            ) + tempUpCross + tempDownCross;
        double normFactor = sumIn / sumCurrent; 
        tempUpCross *= normFactor; 
        tempDownCross *= normFactor; 

        thrust::transform(prStatesNew.begin(), prStatesNew.end(), prStatesNew.begin(), thrust::placeholders::_1 * normFactor);
        thrust::copy(prStatesNew.begin(), prStatesNew.end(), prStates.begin() + (time) * numStates);
        
        std::cout << "UPDATED PRSTATES" << std::endl; 
        for (int i = 0; i < prStates.size(); i++) {
            std::cout << prStates[i] << " "; 
            if ((i + 1) % numStates == 0) { std::cout << std::endl; }
        }

        for (int i = 0; i < 200; i++) {
            std::cout << "=";
        }
        std::cout << std::endl; 


        probUpCrossing[time] = tempUpCross;
        probDownCrossing[time] = tempDownCross;
    }
    double likelihood = 0; 
    if (trial.choice == -1) {
        if (probUpCrossing[numTimeSteps - 1] > 0) {
            likelihood = probUpCrossing[numTimeSteps - 1];
        }
    }
    else if (trial.choice == 1) {
        if (probDownCrossing[numTimeSteps - 1] > 0) {
            likelihood = probDownCrossing[numTimeSteps - 1];
        }
    }
    return likelihood;
}

int main() {

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

            double prob = getTrialLikelihoodGPU(ddm, trial);

            break;
        }
        std::cout << "NLL: " << NLL << std::endl;
    }

    return 0;
}