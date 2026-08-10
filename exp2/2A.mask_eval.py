# Evaluate what happens when we exclude the low producvity grid cells
# 0. Set Up ----------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import fxns
import fxns_analysis as my_fxns
import seaborn as sns
import os
import numpy as np
import xarray as xr
import re

VAR = "npp"

# Prep where to write the analysis figures out to.
FIG_DIR = "./exp2/figs/" + VAR + "/"
DATA_DIR = "./exp2/data/"
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("default")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# 1. Mask Map  ----------------------------------------------------------------------------------------------------
# Quickly plot maps of the masks to get a sense of what grid cells are being dropped when we apply the
# different precentile thresholds.
ALL_TH = [5, 10, 20, 50]

for th in ALL_TH:
    files = [
        os.path.join(DATA_DIR, "mask", f)
        for f in os.listdir(os.path.join(DATA_DIR, "mask"))
        if os.path.isfile(os.path.join(DATA_DIR, "mask", f))
           and re.search(VAR, f)
           and re.search("_" + str(th) + ".nc", f)
    ]

    # Now let's just make some maps of the masks
    d1 = xr.open_dataset(files[0])
    d2 = xr.open_dataset(files[1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    d1_var = d1[VAR]
    d2_var = d2[VAR]

    # handle time dimension if present
    if "time" in d1_var.dims:
        d1_var = d1_var.isel(time=0)
        d2_var = d2_var.isel(time=0)

    d1_var.plot(ax=ax1, cmap="viridis")
    label1 = "r10i1p1f1"
    ax1.set_title(label1)

    d2_var.plot(ax=ax2, cmap="viridis")
    label2 = "r4i1p1f1"
    ax2.set_title(label2)

    fig.suptitle("Historical Boolean Mask, Drop " + str(th) + " Percentile", fontsize=18, y=1.02)

    plt.tight_layout()
    plt.savefig(FIG_DIR + VAR + "_mask_" + str(th) + ".png", dpi=300, bbox_inches="tight")
    # end of the mask level for loop



# 2. Global Comparison  ------------------------------------------------------------------------------------------------
# Compare how the total global carbon values change when we apply the masks.

# Extract the threshold from the file name.
def get_threshold(path):
    # Extract the basename
    s = os.path.basename(path)
    out = os.path.basename(s).rsplit("_", 1)[-1].replace(".nc", "")
    return out


# How much does it matter filtering out the lower values?
# Load the masked files.
global_files = [
    os.path.join(DATA_DIR, "global", f)
    for f in os.listdir(os.path.join(DATA_DIR, "global"))
    if os.path.isfile(os.path.join(DATA_DIR, "global", f))
    and re.search(VAR, f)
]


def multi_partition(text, separators):
    # Join patterns with | (OR)
    regex_pattern = '|'.join(map(re.escape, separators))
    # Lookahead ?= keeps the delimiter on the right part
    # Use ?<= to keep it on the left
    parts = re.split(f'({regex_pattern})', text, maxsplit=1)

    if len(parts) > 1:
        return parts[0], parts[1], parts[2]
    return parts[0], "", ""

# Extract the global totals generated with the different threshold masks.
list_threshold_rslts = []
for f in global_files:

    # Extract meta data
    meta_data = fxns.extrat_exp_ens(f)
    meta_data["mask"] = get_threshold(f)

    # Extract the data and save a data frame.
    ds = xr.open_dataset(f)
    time_array = ds["time"].values
    years = np.array([dt.year for dt in time_array])
    values = ds[VAR].values

    df_data = pd.DataFrame({
        'year': years,
        'value': values,
        'variable': VAR,
    })

    # Combine the meta data and data into a single data frame.
    df_combined = pd.concat([df_data, meta_data], axis=1)
    df_combined = df_combined.ffill()
    list_threshold_rslts.append(df_combined)
    # end of for loop

# Change the list of global results into a single data frame
threshold_df = pd.concat(list_threshold_rslts, axis=0, ignore_index=True)

# Find and load the global files, used in earlier experiments that have not
# filtered out values. Find all the files for the variable of interest to plot.
global_dir = os.path.join("exp0", "data", "global")
to_process = [
    os.path.join(global_dir, f)
    for f in os.listdir(global_dir)
    if re.search(VAR, f)
]

# Extract global values from all the netcdf files
original_global_df = pd.DataFrame()
for f in to_process:

    # Open the nc file.
    ds = xr.open_dataset(f)

    # Extract information
    # Get the ensemble name
    before, sep, after = multi_partition(f, ["historical_", "ssp585_"])
    ENS, sep, after = after.partition("_Lmon")
    year = np.array([date.year for date in ds.time.values])
    value = np.array(ds[VAR].values)
    df = pd.DataFrame({'year': year, "value": value, "variable": VAR, "ensemble": ENS, "experiment": ds.experiment_id})
    original_global_df = pd.concat([original_global_df, df])


# Let's calculate the difference between the original data vs the masked data
original_global_df = original_global_df.rename(columns={"value": "og_values"})
wide_df = pd.merge(threshold_df, original_global_df, on=['year', 'ensemble', 'variable', 'experiment'], how="inner")
wide_df["difference"] = wide_df["og_values"] - wide_df["value"]

# Make the plot to compare how the global values change when the various masks are applied.
# Make the figure
to_plot = wide_df
#to_plot = wide_df[wide_df['year'] < 2020]
plt.figure(figsize=(8, 5))
ax = sns.lineplot(
    data=to_plot,
    x="year",
    y="difference",
    hue="mask",   # or "source", depending on what you want grouped
)

ax.set_ylabel('Change in Global ' + VAR + ' due to masking (Pg C)')
plt.savefig(FIG_DIR + "global_mask_comparison.png", dpi=300, bbox_inches='tight')

# What does the error data look like???
summary_table = to_plot.groupby(['experiment','variable', 'mask'])['difference'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'global_mask_errors'+ VAR + '.csv', float_format='%.2e')



# 4. Error Metric Comparisons ------------------------------------------------------------------------------------------
plt.style.use("default")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

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
    bench_data["mask"] = "0"

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
    old_data["mask"] = "0"

    # Now let's read in all the masked files! that we want to compare with one another.
    DIR = "exp2/data/error_metrics"
    new_files = [
        os.path.join(DIR, f)
        for f in os.listdir(DIR)
        if re.search(V, f)
           and re.search(EM + "1", f)
    ]
    new_data = my_fxns.get_hist_data3(new_files)
    new_data["metric"] = EM
    new_data["source"] = "mask_" + new_data["mask"]

    to_plot = pd.concat([bench_data, old_data, new_data], ignore_index=True)
    to_plot = to_plot[to_plot['value'].notna()].reset_index(drop=True)
    return to_plot

## 4A. MSE -------------------------------------------------------------------------------------------------------------

# Plot the log of the MSE, ideally the density plot would end up being close to the benchmark distribution
to_plot = get_error_histogram_data("mse", VAR)
to_plot["value"] = np.log10(to_plot['value'])

title_name = "EXP2: log " + to_plot.metric[0] + " " + to_plot.variable[0]
metric_name = to_plot.metric[0]
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
plt.savefig(FIG_DIR + "exp2_hist_" + VAR + "_" + metric_name + ".png", dpi=300, bbox_inches='tight')

## 4B. ctbias -------------------------------------------------------------------------------------------------------------

# Plot the log of the MSE, ideally the density plot would end up being close to the benchmark distribution
to_plot = get_error_histogram_data("ctbias", VAR)
to_plot["value"] = np.log10(to_plot['value'])

title_name = "EXP2: log " + to_plot.metric[0] + " " + to_plot.variable[0] + " (~0)"
metric_name = to_plot.metric[0]
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
plt.savefig(FIG_DIR + "exp2_hist_" + VAR + "_" + metric_name + ".png", dpi=300, bbox_inches='tight')

## 4C. ctvar -------------------------------------------------------------------------------------------------------------

# Plot the log of the MSE, ideally the density plot would end up being close to the benchmark distribution
to_plot = get_error_histogram_data("ctvar", VAR)
to_plot["value"] = np.log10(to_plot['value'])

title_name = "EXP2: log " + to_plot.metric[0] + " " + to_plot.variable[0] + " (~1)"
metric_name = to_plot.metric[0]
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
plt.savefig(FIG_DIR + "exp2_hist_" + VAR + "_" + metric_name + ".png", dpi=300, bbox_inches='tight')


## 4D. NS -------------------------------------------------------------------------------------------------------------

# Plot the log of the MSE, ideally the density plot would end up being close to the benchmark distribution
to_plot = get_error_histogram_data("ns", VAR)
to_plot["value"] = np.log10(to_plot['value'])

title_name = "EXP2: log " + to_plot.metric[0] + " " + to_plot.variable[0] + " (<1)"
metric_name = to_plot.metric[0]
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
plt.savefig(FIG_DIR + "exp2_hist_" + VAR + "_" + metric_name + ".png", dpi=300, bbox_inches='tight')

