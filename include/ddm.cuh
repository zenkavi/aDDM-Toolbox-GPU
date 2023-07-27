#ifndef DDM_CUH
#define DDM_CUH

#include <vector> 
#include <string> 
#include <functional> 

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
            DDMTrial *trials, double *likelihoods, int numTrials, 
            float d, float sigma, float barrier, 
            int nonDecisionTime, int timeStep, float approxStateStep, float dec);

        static std::function<double(DDM)> getNLLComputer(std::vector<DDMTrial> trials, std::string computeMethod);
        
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

        static DDM fitModelBasic(std::vector<DDMTrial> trials, std::vector<float> rangeD, std::vector<float> rangeSigma, float barrier, std::string computeMethod="basic");

        static DDM fitModelOptimized(std::vector<DDMTrial> trials, float startD=0.1, float startSigma=1, float deltaD=0.05, float deltaSigma=0.5, float barrier=1, float tolerance=1, std::string computeMethod="basic", std::string optimizer="hooke-jeeves");
};


#endif 