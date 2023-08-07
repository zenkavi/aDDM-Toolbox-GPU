#ifndef MLE_INFO_H
#define MLE_INFO_H

#include <map> 
#include <vector>

template <typename T>
struct MLEinfo {
    T optimal; 
    std::map<T, float> likelihoods; 
};


class ProbabilityData {
    private:
    public: 
        double likelihood; 
        double NLL; 
        std::vector<double> trialLikelihoods; 
        
        ProbabilityData(double likelihood=0, double NLL=0) {
            this->likelihood = likelihood; 
            this->NLL = NLL;
        };
};

#endif