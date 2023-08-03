import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Tuple
import queue

# MapData class is defined to store information for heatmaps
class MapData:
    def __init__(self, par1_sums: defaultdict, par2_sums: defaultdict, par1_label: str, par2_label: str):
        self.par1_sums = par1_sums
        self.par2_sums = par2_sums
        self.par1_label = par1_label
        self.par2_label = par2_label


df = pd.read_csv('results/addm_posteriors.csv')

# num columns - 1 (exclude probability)
N = df.shape[1] - 1

# Calculate the sums of probabilities for each parameter
d_sums = defaultdict(int)
sigma_sums = defaultdict(int)
theta_sums = defaultdict(int) 
for i, row in df.iterrows(): 
    d_val = row['d']
    sigma_val = row['sigma']
    theta_val = row['theta']
    p_val = row['p']
    d_sums[d_val] += p_val
    sigma_sums[sigma_val] += p_val
    theta_sums[theta_val] += p_val

# Queue to store data for bar plots (diagonal subplots)
diags: queue.Queue[defaultdict] = queue.Queue()
diags.put(("d", d_sums))
diags.put(("sigma", sigma_sums))
diags.put(("theta", theta_sums))

# Initialize arrays to store data for heatmaps
d_sigma = np.zeros((len(d_sums), len(sigma_sums)))
d_theta = np.zeros((len(d_sums), len(theta_sums)))
sigma_theta = np.zeros((len(sigma_sums), len(theta_sums)))

# Define a list of MapData objects to store information about each heatmap
heatmaps = [
    MapData(d_sums, sigma_sums, "d", "sigma"), 
    MapData(d_sums, theta_sums, "d", "theta"), 
    MapData(sigma_sums, theta_sums, "sigma", "theta")
]

# Queue to store data for heatmaps
heatmaps_queue: queue.Queue[Tuple[MapData, np.ndarray]] = queue.Queue()

# Iterate over each MapData object and calculate the sums for corresponding heatmap
for map_data in heatmaps: 
    print(map_data.par1_label, map_data.par2_label, "p_sum")
    arr_data = np.zeros((len(map_data.par1_sums), len(map_data.par2_sums)))
    for i in range(len(map_data.par1_sums)):
        for j in range(len(map_data.par2_sums)):
            curr_par1 = list(map_data.par1_sums.keys())[i] # first parameter
            curr_par2 = list(map_data.par2_sums.keys())[j] # second parameter
            for k, row in df.iterrows():
                if row[map_data.par1_label] == curr_par1 and row[map_data.par2_label] == curr_par2:
                    arr_data[i][j] += row["p"]
            print(curr_par1, curr_par2, arr_data[i][j])
    heatmaps_queue.put((map_data, arr_data))
    print()

# Create subplots grid for the bar plots and heatmaps
fig, axes = plt.subplots(figsize=(10, 10), ncols=N, nrows=N)

# Iterate over each subplot and plot either a bar plot or a heatmap
for i in range(N):
    for j in range(N):
        if i < j:
            axes[i, j].axis('off')  # Turn off axes for subplots above the diagonal 
        elif i == j:
            curr = diags.get()
            title = curr[0]
            vals = curr[1]
            g = sns.barplot(x=list(vals.keys()), y=list(vals.values()), ax=axes[i, j], color="grey")
            g.bar_label(g.containers[0], size=8)  # Add labels to the bar plot bars

            keys_labels = list(vals.keys())
            axes[i, j].set_xticks(np.arange(len(keys_labels)))
            axes[i, j].set_xticklabels(keys_labels)
        else:
            curr_heatmap = heatmaps_queue.get()
            data = curr_heatmap[0]
            arr = curr_heatmap[1]
            g = sns.heatmap(arr.T, annot=True, cbar=False, cmap="crest", ax=axes[i, j])
            g.invert_yaxis()  # Invert y-axis for heatmap to match the order of values

            g.set_xticks(np.arange(len(data.par1_sums)) + 0.5, minor=False)  
            g.set_xticklabels(list(data.par1_sums.keys()), minor=False)  # Center the x-axis ticks for the heatmap
            g.set_yticks(np.arange(len(data.par2_sums)) + 0.5, minor=False)  
            g.set_yticklabels(list(data.par2_sums.keys()), minor=False)  # Center the y-axis ticks for the heatmap

# Set labels for x and y axes of plot
axes[0, 0].set_ylabel("d", size=16)
axes[1, 0].set_ylabel("sigma", size=16)
axes[2, 0].set_ylabel("theta", size=16)
axes[2, 0].set_xlabel("d", size=16)
axes[2, 1].set_xlabel("sigma", size=16)
axes[2, 2].set_xlabel("theta", size=16)

plt.tight_layout()
plt.show()
