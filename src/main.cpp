#include <iostream>
#include "addm.h"
#include "ddm.h"
#include "util.h"

int main() {
    std::map<int, std::vector<aDDMTrial>> data = loadDataFromCSV("data/expdata.csv", "data/fixations.csv");
    FixationData fixationData = getEmpiricalDistributions(data);
    aDDM addm = aDDM(0.005f, 0.07f, 1.0f, 1.0f);
    aDDMTrial at = addm.simulateTrial(10, 10, fixationData);
    aDDMexportData(addm, at);
    double p = addm.getTrialLikelihood(at, true);
}