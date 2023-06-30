#ifndef DDM_H
#define DDM_H

#include <vector> 

class DDMTrial {
    private:
    public:
        unsigned int RT;
        int choice;
        float valueLeft;
        float valueRight;
        std::vector<float>RDVs;
        int timeStep;

        DDMTrial(unsigned int RT, int choice, float valueLeft, float valueRight);

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

        DDMTrial simulateTrial(float ValueLeft, float ValueRight, int timeStep=10);
};

#endif 