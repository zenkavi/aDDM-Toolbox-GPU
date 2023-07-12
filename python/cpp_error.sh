#!/bin/bash 

if [ $1 = "addm" ]; then 
    make sim 
    bin/addm_sim
    make nll
    bin/addm_nll
    cp results/addm_simulations.csv ~/aDDM-Toolbox/addm_simulations.csv
    cd ~/aDDM-Toolbox 
    python3 bin/addm_mla_test
    cd ~/aDDM-Toolbox-GPU
    cp ~/aDDM-Toolbox/prob_out.csv results/python_addm_sim_prob.csv
    python3 python/cpp_error.py addm 
else
    make sim 
    bin/ddm_sim 
    make nll 
    bin/ddm_nll 
    cp results/ddm_simulations.csv ~/aDDM-Toolbox/ddm_simulations.csv
    cd ~/aDDM-Toolbox
    python3 bin/ddm_mla_test
    cd ~/aDDM-Toolbox-GPU
    cp ~/aDDM-Toolbox/prob_out.csv results/python_ddm_sim_prob.csv
    python3 python/cpp_error.py
fi
