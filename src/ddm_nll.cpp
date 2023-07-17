#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "ddm.h"
#include "util.h"

float d = 0.005;
float sigma = 0.07;
int barrier = 1;
int valueLeft = 3; 

struct params {
    double NLL; 
    double d; 
    double sigma; 
};

int main() {
    std::vector<DDMTrial> trials; 
    std::ifstream file("results/ddm_simulations.csv");
    std::string line;
    std::getline(file, line);
    int choice; 
    int RT; 
    int valDiff;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::getline(ss, field, ',');
        choice = std::stoi(field);
        std::getline(ss, field, ',');
        RT = std::stoi(field);
        std::getline(ss, field, ',');
        valDiff = std::stoi(field);
        DDMTrial dt = DDMTrial(RT, choice, valueLeft, valueLeft - valDiff);
        trials.push_back(dt);
    }
    file.close();
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    DDM ddm = DDM(d, sigma, barrier);

    std::ofstream fp;
    fp.open("results/ddm_simulations_prob.csv");
    fp << "choice,RT,p\n";

    double NLL = 0; 
    for (DDMTrial dt : trials) {
        double prob = ddm.getTrialLikelihood(dt);
        fp << dt.choice << "," << dt.RT << "," << prob << "\n";
        NLL += -log(prob);
    }
    fp.close();
    std::cout << "NLL: " << NLL << std::endl;
}