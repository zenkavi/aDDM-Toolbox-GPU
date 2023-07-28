#ifndef ADDM_CUH
#define ADDM_CUH

#include <string>
#include <vector>
#include <map>
#include "ddm.cuh"

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

class aDDMTrial: public DDMTrial {
    private:
    public:
        vector<int> fixItem;
        vector<int> fixTime;
        vector<float> fixRDV;
        float uninterruptedLastFixTime;
        std::vector<float>RDVs;

        aDDMTrial(
            unsigned int RT, int choice, int valueLeft, int valueRight, 
            vector<int> fixItem={}, vector<int> fixTime={}, 
            vector<float> fixRDV={}, float uninterruptedLastFixTime=0);

        aDDMTrial() {};
};

class aDDM: public DDM {
    private:
        void callGetTrialLikelihoodKernel(
            bool debug, int trialsPerThread, int numBlocks, int threadsPerBlock, 
            aDDMTrial *trials, double *likelihoods, int numTrials, 
            float d, float sigma, float theta, float barrier, 
            int nonDecisionTime, int timeStep, float approxStateStep, float decay);

    public: 
        float theta;

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
     * @brief Compute the likelihood of the data for a single trial given aDDM Parameters.
     * 
     * @param trial aDDMTrial object.
     * @param debug True if state variables should be printed for debugging purposes
     * @param timeStep value in milliseconds used for binning the time axis.
     * @param approxstateStep used for binning the RDV axis.
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
     * @param fixationDist distribution of the fixation data being used
     * @param timeBins predetermined time bins as used in the fixationDist
     * @return aDDMTrial resulting from the simulation
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
     * @param timeStep Value of in milliseconds used for binning time axis. 
     * @param approxStateStep Used for binning the RDV axis.
     * @return double representing the sum of negative log likelihoods for each trial. 
     */
    double computeParallelNLL(
        vector<aDDMTrial> trials, bool debug=false, int timeStep=10, 
        float approxStateStep=0.1
    );

    /**
     * @brief Compute the total Negative Log Likelihood (NLL) for a vector of aDDMTrials. Use the
     * GPU to maximize the number of trials being computed in parallel. 
     * 
     * @param trials Vector of aDDMTrials that the model should calculcate the NLL for. 
     * @param debug Boolean specifying if state variables should be printed for debugging purposes.
     * @param trialsPerThread Number of trials that each thread should be designated to compute. 
     * Must be divisible by the total number of trials. 
     * @param timeStep Value of in milliseconds used for binning time axis. 
     * @param approxStateStep Used for binning the RDV axis.
     * @return double representing the sum of negative log likelihoods for each trial. 
     */
    double computeGPUNLL(
        vector<aDDMTrial> trials, bool debug=false, int trialsPerThread=10, 
        int timeStep=10, float approxStateStep=0.1
    );

    /**
     * @brief Complete a grid-search based Maximum Likelihood Estimation of all possible parameter 
     * combinations (d, theta, sigma) to determine which parameters are most likely to generate 
     * the provided aDDMTrials. Each potential model generates an NLL value for the dataset and the
     * method returns the model with the minimum NLL value. 
     * 
     * @param trials Vector of aDDMTrials that each model should calculate the NLL for. 
     * @param rangeD Vector of floats representing possible values of d to test for. 
     * @param rangeSigma vector of floats representing possible values of sigma to test for. 
     * @param rangeTheta Vector of floats representing possibel values of theta to test for. 
     * @param barrier Positive magnitude of the signal thresholds.
     * @param computeMethod Computation method to calculate the NLL for each possible models. 
     * Allowed values are {basic, thread, gpu}. "basic" will compute each trial likelihood in
     * parallel and compute the NLL as the sum of all negative log likelihoods. "thread" will
     * use a thread pool to divide all trials into the maximum number of CPU threads and compute
     * the NLL of each block in parallel. "gpu" will call of CUDA kernel to compute the likelihood
     * of each trial in parallel on the GPU. 
     * @return aDDM representing the most optimal model with the lowest total NLL value. 
     */
    static aDDM fitModelMLE(
        vector<aDDMTrial> trials, vector<float> rangeD, vector<float> rangeSigma, 
        vector<float> rangeTheta, float barrier, string computeMethod="basic"
    );
};

#endif 