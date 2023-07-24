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

        DDMTrial() {};
};

class DDM {
    private:
        void callGetTrialLikelihoodKernel(
            bool debug, int trialsPerThread, int numBlocks, int threadsPerBlock, 
            DDMTrial *trials, double *likelihoods, 
            int numTrials, float d, float sigma, float barrier, 
            int nonDecisionTime, int timeStep, float approxStateStep, float dec);
        
    public: 
        float d; 
        float sigma; 
        float barrier; 
        unsigned int nonDecisionTime;
        float bias;

        DDM(float d, float sigma, float barrier, unsigned int nonDecisionTime=0, float bias=0);

        DDM() {}

        double getTrialLikelihood(DDMTrial trial, bool debug=false, int timeStep=10, float approxStateStep=0.1);

        DDMTrial simulateTrial(int ValueLeft, int ValueRight, int timeStep=10);

        double computeParallelNLL(std::vector<DDMTrial> trials, bool debug=false, int timeStep=10, float approxStateStep=0.1);

        double computeGPUNLL(std::vector<DDMTrial> trials, bool debug=false, int trialsPerThread=10, int timeStep=10, float approxStateStep=0.1);
};


#endif 