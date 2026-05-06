import matplotlib.pyplot as plt
import pandas as pd

import fxns_analysis as my_fxns
import seaborn as sns
import os
import numpy as np
import xarray as xr
import re

plt.style.use("default")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


EXP_DIR = "./exp0/"
DATA_DIR = EXP_DIR + 'data/'
FIG_DIR = EXP_DIR + "figs/"

os.makedirs(FIG_DIR, exist_ok=True)



# NPP -------------------------------------------------------------------------------------------------------
# Global ----------------------------------------------------------------------------------------------------
def multi_partition(text, separators):
    # Join patterns with | (OR)
    regex_pattern = '|'.join(map(re.escape, separators))
    # Lookahead ?= keeps the delimiter on the right part
    # Use ?<= to keep it on the left
    parts = re.split(f'({regex_pattern})', text, maxsplit=1)

    if len(parts) > 1:
        return parts[0], parts[1], parts[2]
    return parts[0], "", ""


# Make an empty data frame to save the results in
data = pd.DataFrame()

# Find the historical global files to take a look at
nc_files = [f for f in os.listdir(DATA_DIR + "global") if f.endswith(".nc")]

for f in nc_files:
    if "rh" in f:
        VAR = "rh"
    else:
        VAR = "npp"

    # Get the ensemble name
    before, sep, after = multi_partition(f, ["historical_", "ssp585_"])
    ENS, sep, after = after.partition("_Lmon")

    ds = xr.open_dataset(DATA_DIR + "global/" + f)

    year = np.array([date.year for date in ds.time.values])
    value = np.array(ds[VAR].values)
    df = pd.DataFrame({'year': year, "value": value, "variable": VAR, "ensemble": ENS})
    data = pd.concat([data, df])

data["source"] = "CanESM5 " + data["ensemble"]
data = data[data['variable'] == 'npp']
data["experiment"] = "ssp585"
data.loc[data["year"] < 2020, "experiment"] = "historical"

plt.figure(figsize=(8, 5))
ax = sns.lineplot(
    data=data,
    x="year",
    y="value",
    hue="variable",   # or "source", depending on what you want grouped
    style="source"    # optional: adds line styles
)

ax.set_ylabel('Global NPP Pg C')
plt.savefig(FIG_DIR + "global_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()

summary_table = data.groupby(['experiment', 'ensemble'])['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'global_npp.csv', float_format='%.2e')

# What is the range of the grid cell values historically & in the future???
files = [DATA_DIR + "field/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_field.nc",
         DATA_DIR + "field/CCCma_CanESM5_historical_r4i1p1f1_Lmon_npp_gn_v20190429_field.nc",
         DATA_DIR + "field/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_field.nc",
         DATA_DIR + "field/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_field.nc"]
field_data = my_fxns.get_hist_data(files)
field_data = field_data.dropna()

to_plot = field_data
to_plot['exp_en'] = to_plot['experiment'] + '_' +  to_plot['ensemble']
title_name = to_plot.metric[0] + " " + to_plot.variable[0] + " (Pg C)"
metric_name = to_plot.metric[0]

sns.displot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=100,
    multiple="layer",
    col="experiment"
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "grid_npp_hist.png", dpi=300, bbox_inches='tight')
plt.show()

summary_table = field_data.groupby(['experiment', 'ensemble'])['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'grid_npp.csv', float_format='%.2e')



# Now let's just make some maps of the Mean grid cell
d1 = xr.open_dataset(DATA_DIR + "field/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_field.nc")
d2 = xr.open_dataset(DATA_DIR + "field/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_field.nc")

# Since we are interested in the RMSE we need to take the sqrt here
d1 = d1.mean(dim='time')
d2 = d2.mean(dim='time')

# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'historical r10i1p1f1'
label2 = 'ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP Mean (PgC)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "self_Mean_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()






# RMSE ----------------------------------------------------------------------------------------------------
# First let's take a look at the MSE
files = [DATA_DIR + "error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_mse.nc",
         DATA_DIR + "error_metrics/CCCma_CanESM5_historical_r4i1p1f1_Lmon_npp_gn_v20190429_mse.nc",
         DATA_DIR + "error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_mse.nc",
         DATA_DIR + "error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_mse.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
to_plot = my_fxns.get_hist_data(files)
to_plot["exp_en"] = to_plot["experiment"].astype(str) + " " + to_plot["ensemble"].astype(str)
to_plot["value"] = to_plot["value"] ** (1/2)
to_plot["metric"] = "RMSE"

title_name = to_plot.metric[0] + " " + to_plot.variable[0] + " (Pg C)"
metric_name = to_plot.metric[0]

sns.histplot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=100,
    multiple="layer",
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "self_RMSE_npp.png", dpi=300, bbox_inches='tight')
plt.show()

summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'self_RMSE_npp.csv', float_format='%.2e')


# Now let's just make some maps of the RMSE
d1 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_mse.nc")
d2 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_mse.nc")

# Since we are interested in the RMSE we need to take the sqrt here
d1 = d1 ** 0.5
d2 = d2 ** 0.5

# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'historical r10i1p1f1'
label2 = 'ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP RMSE (PgC)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "self_RMSE_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()



# NS ----------------------------------------------------------------------------------------------------
# Now let's take a look at the NS
files = [DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_ns.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r4i1p1f1_Lmon_npp_gn_v20190429_ns.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_ns.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_ns.nc"]

to_plot = my_fxns.get_hist_data(files)
to_plot["exp_en"] = to_plot["experiment"].astype(str) + " " + to_plot["ensemble"].astype(str)
to_plot["metric"] = "NS (<1)"

title_name = to_plot.metric[0] + " " + to_plot.variable[0]
metric_name = to_plot.metric[0]

sns.histplot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=100,
    multiple="layer",
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "self_NS_npp.png", dpi=300, bbox_inches='tight')
plt.show()

summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'self_NS_npp.csv', float_format='%.2e')


# Now let's just make some maps of the NA
d1 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_ns.nc")
d2 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_ns.nc")

# Since we are interested in the NS we need to take the sqrt here
# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'historical r10i1p1f1'
label2 = 'ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP NS (<1)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "self_NS_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()



# NS -Antartica ----------------------------------------------------------------------------------------------------
# Now let's take a loko at the NS
files = [DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_ns.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r4i1p1f1_Lmon_npp_gn_v20190429_ns.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_ns.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_ns.nc"]

to_plot = my_fxns.get_hist_data_antiantartica(files)
to_plot["exp_en"] = to_plot["experiment"].astype(str) + " " + to_plot["ensemble"].astype(str)
to_plot["metric"] = "NS (<1)"

title_name = to_plot.metric[0] + " " + to_plot.variable[0]
metric_name = to_plot.metric[0]

sns.histplot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=100,
    multiple="layer",
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "self_NS_npp_drop_antartica.png", dpi=300, bbox_inches='tight')
plt.show()

summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'self_NS_npp_drop_antartic.csv', float_format='%.2e')


# Now let's just make some maps of the NA
d1 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_ns.nc")
d2 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_ns.nc")
d1 = d1.where(d1.lat > -60)
d2 = d2.where(d2.lat > -60)


# Since we are interested in the NS we need to take the sqrt here
# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'historical r10i1p1f1'
label2 = 'ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP NS (<1)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "self_NS_npp_map_drop_antartic.png", dpi=300, bbox_inches='tight')
plt.show()




# ctbias ----------------------------------------------------------------------------------------------------
# Now let's take a loko at the ctbias
files = [DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_ctbias.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r4i1p1f1_Lmon_npp_gn_v20190429_ctbias.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_ctbias.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_ctbias.nc"]

to_plot = my_fxns.get_hist_data_antiantartica(files)
to_plot["exp_en"] = to_plot["experiment"].astype(str) + " " + to_plot["ensemble"].astype(str)
to_plot["metric"] = "ctbias (~0)"

title_name = to_plot.metric[0] + " " + to_plot.variable[0]
metric_name = to_plot.metric[0]

sns.histplot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=100,
    multiple="layer",
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "self_ctbias_npp.png", dpi=300, bbox_inches='tight')
plt.show()

summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'self_ctbias_npp.csv', float_format='%.2e')


# Now let's just make some maps of the NA
d1 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_ctbias.nc")
d2 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_ctbias.nc")
d1 = d1.where(d1.lat > -60)
d2 = d2.where(d2.lat > -60)


# Since we are interested in the ctbias we need to take the sqrt here
# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'historical r10i1p1f1'
label2 = 'ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP ctbias (~0)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "self_ctbias_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()

# ctvar ----------------------------------------------------------------------------------------------------
# Now let's take a loko at the ctbias
files = [DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_ctvar.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r4i1p1f1_Lmon_npp_gn_v20190429_ctvar.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_ctvar.nc",
         DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_ctvar.nc"]

to_plot = my_fxns.get_hist_data_antiantartica(files)
to_plot["exp_en"] = to_plot["experiment"].astype(str) + " " + to_plot["ensemble"].astype(str)
to_plot["metric"] = "ctvar (~1)"

title_name = to_plot.metric[0] + " " + to_plot.variable[0]
metric_name = to_plot.metric[0]

sns.histplot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=100,
    multiple="layer",
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "self_ctvar_npp.png", dpi=300, bbox_inches='tight')
plt.show()

summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'self_ctvar_npp.csv', float_format='%.2e')


# Now let's just make some maps of the NA
d1 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_historical_r10i1p1f1_Lmon_npp_gn_v20190429_ctvar.nc")
d2 = xr.open_dataset(DATA_DIR+"error_metrics/CCCma_CanESM5_ssp585_r10i1p1f1_Lmon_npp_gn_v20190429_ctvar.nc")
d1 = d1.where(d1.lat > -60)
d2 = d2.where(d2.lat > -60)


# Since we are interested in the ctbias we need to take the sqrt here
# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'historical r10i1p1f1'
label2 = 'ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP ctvar (~1)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "self_ctvar_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()







