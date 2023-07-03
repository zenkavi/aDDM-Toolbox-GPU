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

        aDDMTrial(
            unsigned int RT, int choice, int valueLeft, int valueRight, 
            vector<int> fixItem={}, vector<int> fixTime={}, 
            vector<float> fixRDV={}, float uninterruptedLastFixTime=0);
};

class aDDM: public DDM {
    private:
    public: 
        float theta;

    aDDM(
        float d, float sigma, float theta, float barrier, 
        unsigned int nonDecisionTime=0, float bias=0
    );

    aDDMTrial simulateTrial(
        int valueLeft, int valueRight, FixationData fixationData, int timeStep=10, 
        int numFixDists=3, fixDists fixationDist={}, vector<int> timeBins={}
    );
};


#endif 