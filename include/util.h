#include <vector>
#include <map>
#include <string>
#include "addm.h"

extern float SEED;
extern vector<string> validFixDistTypes;

struct EXPEntry {
    int parcode;
    int trial;
    int rt;
    int choice;
    int item_left;
    int item_right;
    int valid;
};

struct FIXEntry {
    int parcode;
    int trial;
    int fix_item;
    int fix_time;
};

std::map<int, std::vector<aDDMTrial>> loadDataFromCSV(
    std::string expDataFilename,
    std::string fixDataFilename);

FixationData getEmpiricalDistributions(
    std::map<int, std::vector<aDDMTrial>> data, 
    int timeStep=10, int MaxFixTime=3000,
    int numFixDists=3, std::string fixDistType="fixation",
    std::vector<int> valueDiffs={-3,-2,-1,0,1,2,3},
    std::vector<int> subjectIDs={},
    bool useOddTrials=true, 
    bool useEvenTrials=true, 
    bool useCisTrials=true, 
    bool useTransTrials=true
    );