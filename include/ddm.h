#ifndef DDM_H
#define DDM_H

#include <vector> 

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

        DDMTrial() {}
};

class DDM {
    private:
    public: 
        float d; 
        float sigma; 
        float barrier; 
        unsigned int nonDecisionTime;
        float bias;

        DDM(float d, float sigma, float barrier, unsigned int nonDecisionTime=0, float bias=0);

        DDM() {}

        double getTrialLikelihood(DDMTrial trial, int timeStep=10, float approxStateStep=0.1);

        DDMTrial simulateTrial(int ValueLeft, int ValueRight, int timeStep=10);
};

#endif 