#include <iostream>
#include "addm.h"
#include "ddm.h"
#include "util.h"

int main() {
    std::map<int, std::vector<aDDMTrial>> data = loadDataFromCSV("data/exp.csv", "data/fix.csv");
    FixationData fixationData = getEmpiricalDistributions(data);
    aDDM addm = aDDM(0.005f, 0.07f, 1.0f, 1.0f);
    aDDMTrial at = addm.simulateTrial(10, 10, fixationData);
    std::cout << "choice: " << at.choice << std::endl;
    std::cout << "RT: " << at.RT << std::endl;
    aDDMexportData(addm, at);

    // DDM ddm = DDM(0.005f, 0.065f, 1.0f); 
    // DDMTrial dt = ddm.simulateTrial(10, 10);
    // DDMexportData(ddm, dt);
}