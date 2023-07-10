#ifndef ADDM_H
#define ADDM_H

#include <string>
#include <vector>
#include <map>
#include "ddm.h"

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
};

class aDDM: public DDM {
    private:
    public: 
        float theta;

    /**
     * @brief Construct a new aDDM object.
     * 
     * @param d drift rate.
     * @param sigma noise or standard deviation for the normal distribution.
     * @param theta ranges on [0,1] and indicates level of attentional bias.
     * @param barrier positive magnitude of the signal thresholds. 
     * @param nonDecisionTime amount of time in milliseconds in which only noise 
     * is added to the decision variable. 
     * @param bias corresponds to the initial value of the decision variable. Must 
     * be smaller than barrier. 
     */
    aDDM(
        float d, float sigma, float theta, float barrier, 
        unsigned int nonDecisionTime=0, float bias=0
    );

    /**
     * @brief Compute the likelihood of the data for a single trial given aDDM Parameters.
     * 
     * @param trial aDDMTrial object.
     * @param timestep value in milliseconds used for binning the time axis.
     * @param approxstateStep used for binning the RDV axis.
     * @return double representing the likelihood for the given trial. 
     */
    double getTrialLikelihood(aDDMTrial trial, int timestep=10, float approxstateStep=0.1);

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
};


#endif 