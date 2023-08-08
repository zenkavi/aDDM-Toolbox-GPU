#ifndef ADDM_CUH
#define ADDM_CUH

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include "ddm.cuh"
#include "mle_info.h"

using namespace std;

using fixDists = map<int, vector<float>>;

class FixationData {
    private:
    public:
        string fixDistType;
        float probFixLeftFirst;
        vector<int> latencies;
        vector<int> transitions;
        fixDists fixations;

        FixationData(
            float probFixLeftFirst, vector<int> latencies, 
            vector<int> transitions, fixDists fixations, 
            string fixDistType
        );
    };

/**
 * @brief Implementation of a single aDDMTrial object. 
 * 
 * ADD MORE DOCUMENTATION. 
 *  
 */
class aDDMTrial: public DDMTrial {
    private:
    public:
        vector<int> fixItem; /**< Vector of integers representing the items fixated on during the 
            trial in chronological order. 1 corresponds to left, 2 corresponds to right, and any
            other value is considered a fixation or blank fixation. */
        vector<int> fixTime; /**< Vector of integers corresponding to the duration of each 
            fixation. */
        vector<float> fixRDV; /**< Vector of floats corresponding to the RDV values at the end of
            each fixation. */
        float uninterruptedLastFixTime; /**< Integer corresponding to the duration (milliseconds) 
            that the last fixation in the trial would have if it had not been terminated when a 
            decision had been made. */

        /**
         * @brief Construct a new aDDM Trial object.
         * 
         * @param RT Response time in milliseconds. 
         * @param choice Either -1 (for left item) or +1 (for right item).
         * @param valueLeft Value of the left item. 
         * @param valueRight Value of the right item. 
         * @param fixItem Vector of integers representing the items fixated on during the trial in
         * chronological order. 1 corresponds to left, 2 corresponds to right, and any other value
         * is considered a transition or blank fixation. 
         * @param fixTime Vector of integers corresponding to the duration of each fixation. Must 
         * be equal in size to fixItem. 
         * @param fixRDV Vector of floats corresopnding to the RDV values at the end of each 
         * fixation. 
         * @param uninterruptedLastFixTime Integer corresponding to the duration (milliseconds) 
         * that the last fixation in the trial would have if it had not been terminated when a 
         * decision had been made. 
         */
        aDDMTrial(
            unsigned int RT, int choice, int valueLeft, int valueRight, 
            vector<int> fixItem={}, vector<int> fixTime={}, 
            vector<float> fixRDV={}, float uninterruptedLastFixTime=0);

        /**
         * @brief Construct an empty aDDMTrial object. 
         * 
         */
        aDDMTrial() {};

        static void writeTrialsToCSV(vector<aDDMTrial> trials, string filename);

        static vector<aDDMTrial> loadTrialsFromCSV(string filename);
};

/**
 * @brief Implementation of the attentional Drift Diffusion Model (aDDM). 
 * 
 * ADD MORE DETAILED DESCRIPTION HERE
 * 
 */
class aDDM: public DDM {
    private:
#ifndef EXCLUDE_CUDA_CODE
        void callGetTrialLikelihoodKernel(
            bool debug, int trialsPerThread, int numBlocks, int threadsPerBlock, 
            aDDMTrial *trials, double *likelihoods, int numTrials, 
            float d, float sigma, float theta, float barrier, 
            int nonDecisionTime, int timeStep, float approxStateStep, float decay);
#endif 

    public: 
        float theta; /**< Float between 0 and 1, parameter of the model which 
            controls the attentional bias.*/

        bool operator <( const aDDM &rhs ) const { return (d + sigma + theta < rhs.d + rhs.sigma + rhs.theta); }
        
        /**
         * @brief Construct a new aDDM object.
         * 
         * @param d Drift rate.
         * @param sigma Noise or standard deviation for the normal distribution.
         * @param theta Ranges on [0,1] and indicates level of attentional bias.
         * @param barrier Positive magnitude of the signal thresholds. 
         * @param nonDecisionTime Amount of time in milliseconds in which only noise 
         * is added to the decision variable. 
         * @param bias Corresponds to the initial value of the decision variable. Must 
         * be smaller than barrier. 
         */
        aDDM(
            float d, float sigma, float theta, float barrier, 
            unsigned int nonDecisionTime=0, float bias=0
        );

        /**
         * @brief Construct an empty aDDM object. 
         * 
         */
        aDDM() {}

        /**
         * @brief Compute the likelihood of the trial results provided the current parameters.
         * 
         * @param trial aDDMTrial object.
         * @param debug Boolean sepcifying if state variables should be printed for debugging
         * purposes.
         * @param timeStep Value in milliseconds used for binning the time axis.
         * @param approxstateStep Used for binning the RDV axis.
         * @return double representing the likelihood for the given trial. 
         */
        double getTrialLikelihood(aDDMTrial trial, bool debug=false, 
            int timeStep=10, float approxStateStep=0.1);

        /**
         * @brief Generate simulated fixations provided item values and empirical fixation data. 
         * 
         * @param valueLeft value of the left item
         * @param valueRight value of the right item
         * @param fixationData instance of a FixationData object containing empirical fixation data
         * @param timeStep value of in milliseconds used for binning time axis. 
         * @param numFixDists number of expected fixations in a given trial 
         * @param fixationDist distribution of the fixation data being used. 
         * @param timeBins predetermined time bins as used in the fixationDist. 
         * @return aDDMTrial resulting from the simulation. 
         */
        aDDMTrial simulateTrial(
            int valueLeft, int valueRight, FixationData fixationData, int timeStep=10, 
            int numFixDists=3, fixDists fixationDist={}, vector<int> timeBins={}
        );

        /**
         * @brief Compute the total Negative Log Likelihood (NLL) for a vector of aDDMTrials. Use CPU
         * multithreading to maximize the number of blocks of trials that can have their respective 
         * NLLs computed in parallel. 
         * 
         * @param trials Vector of aDDMTrials that the model should calculcate the NLL for. 
         * @param debug Boolean specifying if state variables should be printed for debugging purposes.
         * @param timeStep Value in milliseconds used for binning the time axis. 
         * @param approxStateStep Used for binning the RDV axis.
         * @return double representing the sum of negative log likelihoods for each trial. 
         */
        ProbabilityData computeParallelNLL(
            vector<aDDMTrial> trials, bool debug=false, int timeStep=10, 
            float approxStateStep=0.1
        );

#ifndef EXCLUDE_CUDA_CODE
        /**
         * @brief Compute the total Negative Log Likelihood (NLL) for a vector of aDDMTrials. Use the
         * GPU to maximize the number of trials being computed in parallel. 
         * 
         * @param trials Vector of aDDMTrials that the model should calculcate the NLL for. 
         * @param debug Boolean specifying if state variables should be printed for debugging purposes.
         * @param trialsPerThread Number of trials that each thread should be designated to compute. 
         * Must be divisible by the total number of trials. 
         * @param timeStep Value in milliseconds used for binning the time axis. 
         * @param approxStateStep Used for binning the RDV axis.
         * @return double representing the sum of negative log likelihoods for each trial. 
         */
        ProbabilityData computeGPUNLL(
            vector<aDDMTrial> trials, bool debug=false, int trialsPerThread=10, 
            int timeStep=10, float approxStateStep=0.1
        );
#endif 

        /**
         * @brief Complete a grid-search based Maximum Likelihood Estimation of all possible parameter 
         * combinations (d, theta, sigma) to determine which parameters are most likely to generate 
         * the provided aDDMTrials. Each potential model generates an NLL value for the dataset and the
         * method returns the model with the minimum NLL value. 
         * 
         * @param trials Vector of aDDMTrials that each model should calculate the NLL for. 
         * @param rangeD Vector of floats representing possible values of d to test for. 
         * @param rangeSigma Vector of floats representing possible values of sigma to test for. 
         * @param rangeTheta Vector of floats representing possible values of theta to test for. 
         * @param barrier Positive magnitude of the signal threshold. 
         * @param computeMethod Computation method to calculate the NLL for each possible model. 
         * Allowed values are {basic, thread, gpu}. "basic" will compute each trial likelihood 
         * sequentially and compute the NLL as the sum of all negative log likelihoods. "thread" will
         * use a thread pool to divide all trials into the maximum number of CPU threads and compute
         * the NLL of each block in parallel. "gpu" will call a CUDA kernel to compute the likelihood
         * of each trial in parallel on the GPU. 
         * @return 
         */
        static MLEinfo<aDDM> fitModelMLE(
            vector<aDDMTrial> trials, vector<float> rangeD, vector<float> rangeSigma, 
            vector<float> rangeTheta, float barrier, string computeMethod="basic", 
            bool normalizePosteriors=false
        );
};

#endif 