#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "addm.h"
#include "ddm.h"
#include "util.h"

float d = 0.005;
float sigma = 0.07;
float theta = 0.5;
int barrier = 1;
int valueLeft = 3; 

int main() {
    std::vector<aDDMTrial> trials; 
    std::ifstream file("results/addm_simulations.csv");
    std::string line;
    std::getline(file, line);

    int ID;
    int choice; 
    int RT; 
    int valDiff;
    int prevID;
    int fItem;
    int fTime; 
    bool firstIter = true; 
    std::vector<int> fixItem;
    std::vector<int> fixTime; 

    aDDMTrial adt = aDDMTrial();
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::getline(ss, field, ',');
        ID = std::stoi(field);
        std::getline(ss, field, ',');
        choice = std::stoi(field);
        std::getline(ss, field, ',');
        RT = std::stoi(field);
        std::getline(ss, field, ',');
        valDiff = std::stoi(field);
        std::getline(ss, field, ',');
        fItem = std::stoi(field);
        std::getline(ss, field, ',');
        fTime = std::stoi(field);
        if (ID == prevID && !firstIter) {
            adt.fixItem.push_back(fItem);
            adt.fixTime.push_back(fTime);
        } else {
            if (firstIter) {
                firstIter = false; 
            } else {
                trials.push_back(adt);
            }
            adt = aDDMTrial(RT, choice, valueLeft, valueLeft - valDiff);
            adt.fixItem.push_back(fItem);
            adt.fixTime.push_back(fTime);

        }
        prevID = ID;
    }
    trials.push_back(adt);
    file.close();
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    aDDM addm = aDDM(d, sigma, theta, barrier);

    std::ofstream fp;
    fp.open("results/addm_simulations_prob.csv");
    fp << "choice,RT,p\n";

    double NLL = 0; 
    for (aDDMTrial adt : trials) {
        double prob = addm.getTrialLikelihood(adt);
        fp << adt.choice << "," << adt.RT << "," << prob << "\n";
        NLL += -log(prob);
    }
    fp.close();
    std::cout << "NLL: " << NLL << std::endl;
}