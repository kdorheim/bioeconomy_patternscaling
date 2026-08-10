# 0. Set Up ----------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import fxns_analysis as my_fxns
import seaborn as sns
import os
import numpy as np
import re

VAR = "rh"
EXP_N = "exp3"

# Prep where to write the analysis figures out to.
FIG_DIR = "./" + EXP_N + "/figs/" + VAR + "/"
DATA_DIR = "./" + EXP_N + "/data/"
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("default")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


# 1. Helper Function ------------------------------------------------------------------

# helper function that laods and formats some error metrics for easy plotting
def get_error_histogram_data(EM, V):
    # Find the banchmark file, aka the best we could posisbly do...
    DIR = "exp0/data/error_metrics"
    benchmark_files = [
        os.path.join(DIR, f)
        for f in os.listdir(DIR)
        if re.search(V, f)
           and re.search(EM, f)
           and re.search('ssp585', f)
    ][0]

    # Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
    bench_data = my_fxns.get_hist_data([benchmark_files])
    # In order to simplify the plots as much as possible we are going to limit the benchmark
    # data to only a single ensemble memeber.
    bench_data["metric"] = EM
    bench_data["source"] = "benchmark"
    bench_data["exp_en"] = "benchmark"

    # Now we are going to load an example of a pervious error metric that we are wanting to do better than
    # since experiment 1 was the out of sample experiment we no longer have the experiment, again we only
    # need to include one to get an idea of how we have changed...
    DIR = "exp1/data/error_metrics"
    old_files = [
        os.path.join(DIR, f)
        for f in os.listdir(DIR)
        if re.search(V, f)
           and re.search(EM, f)
    ][0]

    # Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
    old_data = my_fxns.get_hist_data2([old_files])
    old_data["metric"] = EM
    old_data["source"] = "old LPS"


    # Now let's read in the current experiment data files.
    DIR = EXP_N+"/data/error_metrics"
    new_files = [
        os.path.join(DIR, f)
        for f in os.listdir(DIR)
        if re.search(V, f)
           and re.search(EM , f)
    ]
    new_data = my_fxns.get_hist_data3(new_files)
    new_data["metric"] = EM
    new_data["source"] = EXP_N + "_" + new_data["ensemble"]

    to_plot = pd.concat([bench_data, old_data, new_data], ignore_index=True)
    to_plot = to_plot[to_plot['value'].notna()].reset_index(drop=True)
    return to_plot


def get_summary_table(df):
    summary_table = df.groupby(['metric', 'source'])['value'].describe()
    summary_table = summary_table[['mean', 'std', 'min', 'max']]
    return summary_table


## 4A. Summary Tabels and Diagnositc Plots  -------------------------------------------------------------------------------------------------------------


for ERR_METRIC in ["mse", "ctbias", "ctvar", "ns"]:
    # Plot the log of the MSE, ideally the density plot would end up being close to the benchmark distribution
    to_plot = get_error_histogram_data(ERR_METRIC, VAR)

    # Write up a summary table of the error metrics
    summary_table = get_summary_table(to_plot)
    summary_table.to_csv(FIG_DIR + EXP_N + '_' + VAR + '_' + ERR_METRIC + '.csv', float_format='%.2e')

    to_plot["value"] = np.log10(to_plot['value'])
    title_name = EXP_N + ": log " + ERR_METRIC + " " + to_plot.variable[0]
    metric_name = ERR_METRIC
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=to_plot,
        x="value",
        hue="source",
        multiple="layer",
        fill=True
    )
    plt.xlabel(metric_name)
    plt.title(title_name)
    plt.savefig(FIG_DIR + EXP_N + "_" + VAR + "_" + ERR_METRIC + ".png", dpi=300, bbox_inches='tight')

















