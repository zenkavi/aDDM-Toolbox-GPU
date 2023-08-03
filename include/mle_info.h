#ifndef MLE_INFO_H
#define MLE_INFO_H

#include <map> 

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
        
        ProbabilityData(double likelihood=0, double NLL=0) {
            this->likelihood = likelihood; 
            this->NLL = NLL;
        };

        operator double() const { return NLL; }
};

#endif