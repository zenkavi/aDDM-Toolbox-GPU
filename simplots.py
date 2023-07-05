import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from typing import Dict, List

def main():
    val_diff_to_rts: Dict[int, List[int]] = {}
    choice_to_rts: Dict[int, List[int]] = {-1 : list(), 1 : list()}
    df = pd.read_csv("results/simulations.csv")
    df = df.reset_index()

    for _, row in df.iterrows():
        choice = row['choice']
        rt = row['RT']
        val_diff = row['valDiff']
        if val_diff in val_diff_to_rts:
            val_diff_to_rts[val_diff].append(rt)
        else:
            val_diff_to_rts[val_diff] = [rt]
        if choice in choice_to_rts:
            choice_to_rts[choice].append(rt)
        else:
            choice_to_rts[choice] = [rt]

    sorted(val_diff_to_rts)
    for vd, rts in val_diff_to_rts.items():
        plt.hist(rts, bins=range(0, 8000, 200), label=vd, edgecolor='black', alpha=0.5)
    
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()