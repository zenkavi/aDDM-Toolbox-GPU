import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import sys 
from datetime import datetime

def main():
    if (len(sys.argv) > 1 and sys.argv[1] == "addm"):
        df1 = pd.read_csv("results/addm_simulations_prob.csv")
        df2 = pd.read_csv("results/python_addm_sim_prob.csv")
    else:
        df1 = pd.read_csv("results/ddm_simulations_prob.csv")
        df2 = pd.read_csv("results/python_ddm_sim_prob.csv")
    cpp_error = list()
    py_error = list()
    for i, row in df1.iterrows():
        cpp_error.append(row["p"])
        py_error.append(df2["p"][i])
    
    plt.figure(figsize=(15, 15))
    plt.scatter(cpp_error, py_error)

    x = np.arange(0, 1, 0.1)
    y = x
    plt.plot(x, y, c="red")
    plt.xlim([0, 0.0175])
    plt.ylim([0, 0.0175])

    plt.xlabel("cpp values")
    plt.ylabel("python values")

    if (len(sys.argv) > 1 and "save" in sys.argv):
        currTime = datetime.now().strftime(u"%Y-%m-%d_%H:%M:%S")
        plt.savefig("imgs/py_cpp_error_" + currTime + ".png")

if __name__=="__main__":
    main()