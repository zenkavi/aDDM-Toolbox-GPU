#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <set>
#include <cmath> 
#include <fstream>
#include <boost/math/distributions/normal.hpp>
#include "nlohmann/json.hpp"
#include "util.h"
#include "addm.h"

using json = nlohmann::json;

float SEED = 100;
float DECAY = 0;
vector<string> validFixDistTypes = {"simple", "difficulty", "fixation"};


double probabilityDensityFunction(float mean, float sigma, float x) {
    boost::math::normal_distribution<double> dist(mean, sigma);
    double pdf = boost::math::pdf(dist, x);
    return pdf; 
}

double cumulativeDensityFunction(float mean, float sigma, float x) {
    boost::math::normal_distribution<double> dist(mean, sigma);
    double cdf = boost::math::cdf(dist, x);
    return cdf;
}

std::map<int, std::vector<aDDMTrial>> loadDataFromCSV(
    std::string expDataFilename, 
    std::string fixDataFilename) {

    // subjectID -> aDDM Trials
    std::map<int, std::vector<aDDMTrial>> data;
    std::map<int, std::vector<EXPEntry>> IDtoEXP;
    std::set<int> subjectIDs;

    std::ifstream expFile(expDataFilename);
    std::vector<EXPEntry> expData;
    std::string eline;
    std::getline(expFile, eline);
    while (std::getline(expFile, eline)) {
        std::stringstream ss(eline);
        std::string field;
        EXPEntry entry;
        std::getline(ss, field, ',');
        entry.parcode = std::stoi(field);
        subjectIDs.insert(entry.parcode);
        std::getline(ss, field, ',');
        entry.trial = std::stoi(field);
        std::getline(ss, field, ',');
        entry.rt = std::stoi(field);
        std::getline(ss, field, ',');
        entry.choice = std::stoi(field);
        std::getline(ss, field, ',');
        entry.item_left = std::stoi(field);
        std::getline(ss, field, ',');
        entry.item_right = std::stoi(field);
        std::getline(ss, field, ',');
        entry.valid = std::stoi(field);
        expData.push_back(entry);
        if (IDtoEXP.count(entry.parcode)) {
            IDtoEXP.at(entry.parcode).push_back(entry);
        } else {
            IDtoEXP.insert({entry.parcode, {}});
        }
    }
    expFile.close();

    for (int subjectID : subjectIDs) {
        data.insert({subjectID, {}});
        std::set<int> trialIDs;
        for (EXPEntry e : expData) {
            if (e.parcode == subjectID) {
                trialIDs.insert(e.trial);
            }
        }
        std::vector<int> dataTrial;
        for (int trialID : trialIDs) {
            for (EXPEntry e : expData) {
                if (e.trial == trialID && e.parcode == subjectID) {
                    data.at(subjectID).push_back(
                        aDDMTrial(e.rt, e.choice, e.item_left, e.item_right)
                    );
                }   
            }
        }
    }
    

    std::ifstream fixFile(fixDataFilename);
    std::vector<FIXEntry> fixData;
    subjectIDs.clear();
    std::string fline;
    std::getline(fixFile, fline); 
    while (std::getline(fixFile, fline)) {
        std::stringstream ss(fline);
        std::string field;
        FIXEntry entry;
        std::getline(ss, field, ',');
        entry.parcode = std::stoi(field);
        subjectIDs.insert(entry.parcode);
        std::getline(ss, field, ',');
        entry.trial = std::stoi(field);
        std::getline(ss, field, ',');
        entry.fix_item = std::stoi(field);
        std::getline(ss, field, ',');
        entry.fix_time = std::stoi(field);
        fixData.push_back(entry);
    }
    fixFile.close();

    for (int subjectID : subjectIDs) {
        if (!data.count(subjectID)) {
            continue;
        }
        std::set<int> trialIDs;
        std::vector<FIXEntry> subjectEntries;
        for (FIXEntry f : fixData) {
            if (f.parcode == subjectID) {
                trialIDs.insert(f.trial);
                subjectEntries.push_back(f);
            }
        }
        int t = 0;
        for (int trialID : trialIDs) {
            std::vector<int> fixItem;
            std::vector<int> fixTime;
            for (FIXEntry fs : subjectEntries) {
                if (fs.trial == trialID) {
                    fixItem.push_back(fs.fix_item);
                    fixTime.push_back(fs.fix_time);
                }
            }
            data.at(subjectID).at(t).fixItem = fixItem;
            data.at(subjectID).at(t).fixTime = fixTime;
            t++;
        }
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

    if (!std::count(
    validFixDistTypes.begin(), validFixDistTypes.end(), fixDistType)) {
        throw std::invalid_argument(
            "Argument type must be one of {simple, difficulty, fixation}");
    }

    int countLeftFirst = 0;
    int countTotalTrials = 0;
    std::vector<int> latencies;
    std::vector<int> transitions;
    std::map<int, std::vector<float>> fixations;
    for (int fn = 1; fn < numFixDists + 1; fn++) {
        if (fixDistType == "simple") {
            // fixations.at(fn) = {};
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
                            if (!fixations.count(fixNumber)) {
                                fixations.insert({fixNumber, {}});
                            }
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

void DDMexportData(DDM ddm, DDMTrial dt) {
    std::ofstream o("results/data.json");
    json j;
    j["d"] = ddm.d;
    j["sigma"] = ddm.sigma;
    j["barrier"] = ddm.barrier;
    j["NDT"] = ddm.nonDecisionTime;
    j["bias"] = ddm.bias;
    j["RT"] = dt.RT;
    j["choice"] = dt.choice;
    j["vl"] = dt.valueLeft;
    j["vr"] = dt.valueRight;
    j["RDVs"] = dt.RDVs;
    j["timeStep"] = dt.timeStep;
    o << std::setw(4) << j << std::endl;        
}

void aDDMexportData(aDDM addm, aDDMTrial adt) {
    std::ofstream o("results/data.json");
    json j;
    j["d"] = addm.d;
    j["sigma"] = addm.sigma;
    j["theta"] = addm.theta;
    j["barrier"] = addm.barrier;
    j["NDT"] = addm.nonDecisionTime;
    j["bias"] = addm.bias;
    j["RT"] = adt.RT;
    j["choice"] = adt.choice;
    j["vl"] = adt.valueLeft;
    j["vr"] = adt.valueRight;
    j["RDVs"] = adt.RDVs;
    j["fixItem"] = adt.fixItem;
    j["fixTime"] = adt.fixTime;
    j["timeStep"] = adt.timeStep;
    o << std::setw(4) << j << std::endl;        
}