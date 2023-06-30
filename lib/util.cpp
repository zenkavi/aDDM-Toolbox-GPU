#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include "util.h"
#include "addm.h"

struct EXPEntry {
    int parcode;
    int trial;
    double rt;
    int choice;
    int item_left;
    int item_right;
    int valid;
};

float SEED = 100;
vector<string> validFixDistTypes = {"simple", "difficulty", "fixation"};

std::map<int, std::vector<aDDMTrial>> loadDataFromCSV(
    std::string expDataFilename, 
    std::string fixDataFilename, 
    bool convertItemValues) {

    std::ifstream file(expDataFilename);
    std::vector<EXPEntry> expData;

    std::string line;
    std::getline(file, line);  // Read and discard the header line

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        EXPEntry entry;

        std::getline(ss, field, ',');
        entry.parcode = std::stoi(field);
        std::getline(ss, field, ',');
        entry.trial = std::stoi(field);
        std::getline(ss, field, ',');
        entry.rt = std::stod(field);
        std::getline(ss, field, ',');
        entry.choice = std::stoi(field);
        std::getline(ss, field, ',');
        entry.item_left = std::stoi(field);
        std::getline(ss, field, ',');
        entry.item_right = std::stoi(field);
        std::getline(ss, field, ',');
        entry.valid = std::stoi(field);
        expData.push_back(entry);
    }

    file.close();

    std::map<int, std::vector<aDDMTrial>> data;
    
    for (const auto& entry : expData) {
        std::cout << "parcode: " << entry.parcode << std::endl;
        std::cout << "trial: " << entry.trial << std::endl;
        std::cout << "rt: " << entry.rt << std::endl;
        std::cout << "choice: " << entry.choice << std::endl;
        std::cout << "item_left: " << entry.item_left << std::endl;
        std::cout << "item_right: " << entry.item_right << std::endl;
        std::cout << "valid: " << entry.valid << std::endl;
        std::cout << "------------------" << std::endl;
    }

    return data;
}


FixationData getEmpiricalDistributions(
    std::map<int, std::vector<aDDMTrial>> data, 
    int timeStep, int MaxFixTime,
    int numFixDists, std::string fixDistType,
    std::vector<int> valueDiffs,
    std::vector<int> subjectIDs,
    bool useOddTrials, 
    bool useEvenTrials, 
    bool useCisTrials, 
    bool useTransTrials) {

    // if (!std::count(
    // validFixDistTypes.begin(), validFixDistTypes.end(), fixDistType)) {
    //     throw std::invalid_argument(
    //         "Argument type must be one of {simple, difficulty, fixation}");
    // }

    int countLeftFirst = 0;
    int countTotalTrials = 0;
    std::vector<int> latencies;
    std::vector<int> transitions;
    std::map<int, std::vector<float>> fixations;
    for (int fn = 1; fn < numFixDists + 1; fn++) {
        if (fixDistType == "simple") {
            fixations.at(fn) = {};
        } else {
            throw std::invalid_argument("simple is not implemented");
        }
    }

    if (subjectIDs.empty()) {
        for (auto i : data) {
            subjectIDs.push_back(i.first);
        }
    }

    for (int subjectID : subjectIDs) {
        int trialID = 0;
        for (aDDMTrial trial : data.at(subjectID)) {
            if (!useOddTrials && trialID % 2 != 0) {
                continue;
            }
            if (!useEvenTrials && trialID % 2 == 0) {
                continue;
            }
            bool isCisTrial = trial.valueLeft * trial.valueRight >= 0 ? true : false;
            bool isTransTrial = trial.valueLeft * trial.valueRight <= 0 ? true : false;
            if (!useCisTrials && isCisTrial && !isTransTrial) {
                continue;
            }
            if (!useTransTrials && isTransTrial && !isCisTrial) {
                continue;
            }
            bool allZero = std::all_of(
                trial.fixItem.begin(), trial.fixItem.end(), 
                [](int i){return i == 0;}
            );
            bool containsOne = std::find(
                trial.fixItem.begin(), trial.fixItem.end(), 1) != trial.fixItem.end();
            bool containsTwo = std::find(
                trial.fixItem.begin(), trial.fixItem.end(), 2) != trial.fixItem.end();
            if (allZero || !(containsOne || containsTwo)) {
                continue;
            }

            std::map<int, int> fixUnfixValueDiffs;
            fixUnfixValueDiffs.insert({1, trial.valueLeft - trial.valueRight});
            fixUnfixValueDiffs.insert({2, trial.valueRight - trial.valueLeft});

            int excludeCount = 0;
            for (int i = trial.fixItem.size() - 1; i >= 0; i--) {
                excludeCount++;
                if (trial.fixItem[i] == 1 || trial.fixItem[i] == 2) {
                    break;
                }
            }

            int latency = 0;
            bool firstItemFixReached = false;
            int fixNumber = 1;
            for (int i = 0; i < trial.fixItem.size() - excludeCount; i++) {
                if (trial.fixItem.at(i) != 1 && trial.fixItem.at(i) != 2) {
                    if (!firstItemFixReached) {
                        latency += trial.fixTime.at(i);
                    } else if (
                        trial.fixTime.at(i) >= timeStep &&
                        trial.fixTime.at(i) <= MaxFixTime
                        ) {
                        transitions.push_back(trial.fixTime.at(i));
                    }
                } else {
                    if (!firstItemFixReached) {
                        firstItemFixReached = true;
                        latencies.push_back(latency);
                    }
                    if (fixNumber == 1) {
                        countTotalTrials++;
                        if (trial.fixItem.at(i) == 1) {
                            countLeftFirst++;
                        }
                    }
                    if (trial.fixTime.at(i) >= timeStep && 
                        trial.fixTime.at(i) <= MaxFixTime
                        ) {
                        if (fixDistType == "simple") {
                            fixations.at(fixNumber).push_back(trial.fixTime.at(i));
                        } else {
                            throw std::invalid_argument("not implemented");
                        }
                    }
                    if (fixNumber < numFixDists) {
                        fixNumber++;
                    }
                }
            }
        }
    }
    float probFixLeftFirst = (float) countLeftFirst / (float) countTotalTrials;
    return FixationData(probFixLeftFirst, latencies, transitions, fixations, fixDistType);
}