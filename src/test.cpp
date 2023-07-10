#include <iostream>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <sstream>
#include "../include/ddm.h"
#include "../include/util.h"

float d = 0.005;
float sigma = 0.07;
int barrier = 1;

int main() {
    std::vector<DDMTrial> trials; 
    std::ifstream file("results/ddm_data.csv");
    std::string line;
    std::getline(file, line);
    int tn;
    int choice; 
    int RT; 
    int valueLeft;
    int valueRight;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::getline(ss, field, ',');
        tn = std::stoi(field);
        std::getline(ss, field, ',');
        RT = std::stoi(field);
        std::getline(ss, field, ',');
        choice = std::stoi(field);
        std::getline(ss, field, ',');
        valueLeft = std::stoi(field);
        std::getline(ss, field, ',');
        valueRight = std::stoi(field);
        DDMTrial dt = DDMTrial(RT, choice, valueLeft, valueRight);
        trials.push_back(dt);
    }
    file.close();
    std::cout << "Counted " << trials.size() << " trials." << std::endl;

    for (auto trial : trials) {
        std::cout << trial.choice << " " << trial.RT << std::endl;
    }

    DDM ddm = DDM(d, sigma, barrier);

    std::ofstream fp;
    fp.open("results/ddm_data_prob.csv");
    fp << "choice,RT,p,valLeft,valRight\n";


    double NLL = 0; 
    for (DDMTrial dt : trials) {
        double prob = ddm.getTrialLikelihood(dt);
        if (prob == 0) {
            prob = pow(10, -20);
        }
        fp << dt.choice << "," << dt.RT << "," << prob << "," << dt.valueLeft << "," << dt.valueRight << "\n";
        NLL += -log(prob);
    }
    std::cout << "NLL: " << NLL << std::endl;
}