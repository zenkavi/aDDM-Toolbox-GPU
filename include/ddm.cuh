#ifndef DDM_CUH
#define DDM_CUH

#include <vector> 
#include <string> 
#include <functional> 
#include <tuple>
#include <map> 
#include "mle_info.h"

class DDMTrial {
    private:
    public:
        unsigned int RT;
        int choice;
        int valueLeft;
        int valueRight;
        std::vector<float>RDVs;
        int timeStep; 

        DDMTrial(unsigned int RT, int choice, int valueLeft, int valueRight);

        DDMTrial() {};

        static void writeTrialsToCSV(std::vector<DDMTrial> trials, std::string filename);

        static std::vector<DDMTrial> loadTrialsFromCSV(std::string filename);
};

class DDM {
    private:
#ifndef EXCLUDE_CUDA_CODE
        void callGetTrialLikelihoodKernel(
            bool debug, 
            int trialsPerThread, int numBlocks, int threadsPerBlock, 
            DDMTrial *trials, double *likelihoods, int numTrials, 
            float d, float sigma, float barrier, 
            int nonDecisionTime, int timeStep, float approxStateStep, float dec);
#endif 
        
    public: 
        float d; /**< Float parameter of the model that controls the speed of integration. Referred
            to as drift rate. */
        float sigma; /**< Float parameter of the model that control noise of the RDV signal and 
            standard deviation for any sampled normal distributions. */
        float barrier; /**< Positive float representing the magnitude of the signal threshold. */
        unsigned int nonDecisionTime; /**< Non-negative integer representing the amount of time in 
            milliseconds in which only noise is added to the decision variable. */
        float bias; /**< Float corresponding to the initial RDV. Must be smaller than the 
            barrier. */

        bool operator <( const DDM &rhs ) const { return (d + sigma < rhs.d + rhs.sigma); }

        /**
         * @brief Construct a new DDM object.
         * 
         * @param d Drift rate. 
         * @param sigma Noise or standard deviation for the normal distribution. 
         * @param barrier Positive magnitude of the signal threshold. 
         * @param nonDecisionTime Amount of time in milliseconds in which only noise is added to 
         * the decision variable. 
         * @param bias Corresponds to the initial value of the decision variable. Must be smaller
         * than the barrier. 
         */
        DDM(float d, float sigma, float barrier, unsigned int nonDecisionTime=0, float bias=0);

        /**
         * @brief Construct an empty DDM object. 
         * 
         */
        DDM() {}

        /**
         * @brief Compute the likelihood of the trial results provided the current parameters. 
         * 
         * @param trial DDMTrial object. 
         * @param debug Boolean specifying if state variables should be printed for debugging 
         * purposes. 
         * @param timeStep Value in milliseconds used for binning the time axis. 
         * @param approxStateStep Used for binning the RDV axis. 
         * @return double representing the likelihood for the given trial. 
         */
        double getTrialLikelihood(
            DDMTrial trial, bool debug=false, int timeStep=10, float approxStateStep=0.1);

        /**
         * @brief Generate a simulated DDM trial provided item values. 
         * 
         * @param ValueLeft Value of the left item. 
         * @param ValueRight Value of the right item. 
         * @param timeStep Value in milliseconds used for binning the time axis. 
         * @return DDMTrial resulting from the simulation. 
         */
        DDMTrial simulateTrial(int ValueLeft, int ValueRight, int timeStep=10);

        /**
         * @brief Compute the total Negative Log Likelihood (NLL) for a vector of DDMTrials. Use 
         * CPU multithreading to maximize the number of blocks of trials that can have their 
         * respective NLLs computed in parallel. 
         * 
         * @param trials Vector of DDMTrials that the model should calculate the NLL for. 
         * @param debug Boolean specifying if state variables should be printed for debugging 
         * purposes. 
         * @param timeStep Value in milliseconds used for binning the time axis. 
         * @param approxStateStep Used for binning the RDV axis. 
         * @return double representing the sum of negative log likelihoods for each trial. 
         */
        ProbabilityData computeParallelNLL(
            std::vector<DDMTrial> trials, bool debug=false, int timeStep=10, float approxStateStep=0.1);

#ifndef EXCLUDE_CUDA_CODE
        /**
         * @brief Compute the total Negative Log Likelihood (NLL) for a vector of DDMTrials. Use
         * the GPU to maximize the number of trials being computed in parallel. 
         * 
         * @param trials Vector of DDMTrials that the model should calculate the NLL for. 
         * @param debug Boolean specifying if state variables should be printed for debugging 
         * purposes. 
         * @param trialsPerThread Number of trials that each thread should be designated to 
         * copmute. Must be divisible by the total number of trials. 
         * @param timeStep Value in milliseconds used for binning the time axis. 
         * @param approxStateStep Used for binning the RDV axis. 
         * @return double representing the sum of negative log likelihoods for each trial. 
         */
        ProbabilityData computeGPUNLL(
            std::vector<DDMTrial> trials, bool debug=false, int trialsPerThread=10, 
            int timeStep=10, float approxStateStep=0.1);
#endif 

        /**
         * @brief Copmlete a grid-search based Maximum Likelihood Estimation of all possible 
         * paramters combinations (d, sigma) to determine which parameters are most likely to 
         * generate the provided DDMTrials. Each potential model generates an NLL value for the 
         * dataset and the method returns the model with the minimum NLL value. 
         * 
         * @param trials Vector of DDMTrials that each model should calculate the NLL for. 
         * @param rangeD Vector of floats representing possible values of d to dest for. 
         * @param rangeSigma Vector of floats representing possible values of sigma to test for. 
         * @param barrier Positive magnitude of the sigmal threshold. 
         * @param computeMethod Computation method to calculate the NLL for each possible model.
         * Allowed values are {basic, thread, gpu}. "basic" will compute each trial likelihood 
         * sequentially and compute the NLL as the sum of all negative log likelihoods. "thread" 
         * will use a thread pool to divide all trials into the maximum number of CPU threads and
         * compute the NLL of each block of trials in parallel. "gpu" will call a CUDA kernel to 
         * compute the likelihood of each trial in parallel on the GPU. 
         * @return 
         */
        static MLEinfo<DDM> fitModelMLE(
            std::vector<DDMTrial> trials, std::vector<float> rangeD, 
            std::vector<float> rangeSigma, float barrier, 
            std::string computeMethod="basic",
            bool normalizePosteriors=false);
};

#endif 