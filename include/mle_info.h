#ifndef MLE_INFO_H
#define MLE_INFO_H

#include <map> 

template <typename T>
struct MLEinfo {
    T optimal; 
    std::map<T, float> posteriors; 
};

#endif