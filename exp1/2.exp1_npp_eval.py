# Eval how well the historical trained pattern scalars to predict future ssp585 results
# 0. Set Up ----------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import fxns_analysis as my_fxns
import seaborn as sns
import os
import numpy as np
import xarray as xr


EXP_DIR = "./exp1/"
DATA_DIR = EXP_DIR + 'data/'
ERROR_DIR = DATA_DIR + "error_metrics/"
FIG_DIR = EXP_DIR + "npp_figs/"

plt.style.use("default")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

os.makedirs(FIG_DIR, exist_ok=True)



# 1. Main Chunk ----------------------------------------------------------------------------------------------------

# RMSE ----------------------------------------------------------------------------------------------------
# First let's take a look at the MSE from exp0
files = ["exp0/data/error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_mse.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
bench_data = my_fxns.get_hist_data_antiantartica(files)
bench_data["exp_en"] = bench_data["experiment"].astype(str) + " " + bench_data["ensemble"].astype(str)
bench_data["value"] = bench_data["value"] ** (1/2)
bench_data["metric"] = "RMSE"
bench_data["source"] = "benchmark"
bench_data["exp_en"] = "benchmark"

files = [ERROR_DIR+"npp_r4i1p1f1_mse1.nc",
         ERROR_DIR+"npp_r4i1p1f1_mse2.nc",
         ERROR_DIR+"npp_r10i1p1f1_mse1.nc",
         ERROR_DIR+"npp_r10i1p1f1_mse2.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
futu_data = my_fxns.get_hist_data2(files)
futu_data["exp_en"] = futu_data["ensemble"].astype(str) +  futu_data["metric"].astype(str)
futu_data["value"] = futu_data["value"] ** (1/2)
futu_data["metric"] = "RMSE"
futu_data["source"] = "LPS"

to_plot = pd.concat([bench_data, futu_data], ignore_index=True)


title_name =  to_plot.metric[0] + " " + to_plot.variable[0] + " (Pg C)"
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
plt.savefig(FIG_DIR + "fut_RMSE_npp.png", dpi=300, bbox_inches='tight')
plt.show()

summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'fut_RMSE_npp.csv', float_format='%.2e')

# Now let's just make some maps of the RMSE
d1 = xr.open_dataset(ERROR_DIR+"npp_r4i1p1f1_mse1.nc")
d2 = xr.open_dataset(ERROR_DIR+"npp_r10i1p1f1_mse2.nc")
d1 = d1.where(d1.lat > -60)
d2 = d2.where(d2.lat > -60)


# Since we are interested in the RMSE we need to take the sqrt here
d1 = d1 ** 0.5
d2 = d2 ** 0.5

# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'pred ssp585 r4i1p1f1'
label2 = 'pred ssp585 r10i1p1f1'

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
plt.savefig(FIG_DIR + "fut_RMSE_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()



# NS ----------------------------------------------------------------------------------------------------
# Now let's take a look at the NS, the benchmark data comes from exp0
files = ["./exp0/data/error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_ns.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
bench_data = my_fxns.get_hist_data_antiantartica(files)
bench_data["exp_en"] = bench_data["experiment"].astype(str) + " " + bench_data["ensemble"].astype(str)
bench_data["metric"] = "NS"
bench_data["source"] = "benchmark"
bench_data["exp_en"] = "benchmark"

files = [ERROR_DIR+"npp_r4i1p1f1_ns1.nc",
         ERROR_DIR+"npp_r4i1p1f1_ns2.nc",
         ERROR_DIR+"npp_r10i1p1f1_ns1.nc",
         ERROR_DIR+"npp_r10i1p1f1_ns2.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
futu_data = my_fxns.get_hist_data2(files)
futu_data["exp_en"] = futu_data["ensemble"].astype(str) + futu_data["metric"].astype(str)
futu_data["metric"] = "NS"
futu_data["source"] = "LPS"

to_plot = pd.concat([bench_data, futu_data], ignore_index=True)
to_plot['value'] = np.log10(to_plot['value'])
title_name =  "log " + to_plot.metric[0] + " " + to_plot.variable[0] + " (<1)"
metric_name = to_plot.metric[0]

sns.histplot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=500,
    multiple="layer",
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "fut_ns_npp.png", dpi=300, bbox_inches='tight')
plt.show()

to_plot = pd.concat([bench_data, futu_data], ignore_index=True)
summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'fut_ns_npp.csv', float_format='%.2e')

# Now let's just make some maps of the NS
d1 = xr.open_dataset(ERROR_DIR+"npp_r4i1p1f1_ns1.nc")
d1 = d1.where(d1.lat > -60)
# find a lat and lon grid cell where we are doing "good" ns <1
#good_points = d1.where(d1 < 1).stack(points=("lat", "lon")).dropna("points")
#coords = list(zip(good_points["lat"].values, good_points["lon"].values))
#point_lat = 26
#point_lon = 360-80
#d1.sel(lat=26, lon=280, method="nearest")['npp'].values


d2 = xr.open_dataset(ERROR_DIR+"npp_r10i1p1f1_ns2.nc")
d2 = d2.where(d2.lat > -60)

# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'pred ssp585 r4i1p1f1'
label2 = 'pred ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    vmax=1,
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    vmax=1,
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP NS (<1)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "fut_NS_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()


# ctbais ----------------------------------------------------------------------------------------------------
# Now let's take a look at the NS, recall the benchamrk data is coming from exp0
files = ["./exp0/data/error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_ctbias.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
bench_data = my_fxns.get_hist_data_antiantartica(files)
bench_data["exp_en"] = bench_data["experiment"].astype(str) + " " + bench_data["ensemble"].astype(str)
bench_data["metric"] = "ctbias"
bench_data["source"] = "benchmark"
bench_data["exp_en"] = "benchmark"

files = [ERROR_DIR+"npp_r4i1p1f1_ctbias1.nc",
         ERROR_DIR+"npp_r4i1p1f1_ctbias2.nc",
         ERROR_DIR+"npp_r10i1p1f1_ctbias1.nc",
         ERROR_DIR+"npp_r10i1p1f1_ctbias2.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
futu_data = my_fxns.get_hist_data2(files)
futu_data["exp_en"] = futu_data["ensemble"].astype(str) + futu_data["metric"].astype(str)
futu_data["metric"] = "ctbias"
futu_data["source"] = "LPS"

to_plot = pd.concat([bench_data, futu_data], ignore_index=True)
to_plot['value'] = np.log10(to_plot['value'])
title_name = "log" + to_plot.metric[0] + " " + to_plot.variable[0] + " (~0)"
metric_name = to_plot.metric[0]

sns.histplot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=500,
    multiple="layer",
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "fut_ctbais_npp.png", dpi=300, bbox_inches='tight')
plt.show()


to_plot = pd.concat([bench_data, futu_data], ignore_index=True)
summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'fut_ctbais_npp.csv', float_format='%.2e')

# Now let's just make some maps of the ctbais
d1 = xr.open_dataset(ERROR_DIR+"npp_r4i1p1f1_ctbias1.nc")
d2 = xr.open_dataset(ERROR_DIR+"npp_r10i1p1f1_ctbias2.nc")
d1 = d1.where(d1.lat > -60)
d2 = d2.where(d2.lat > -60)

# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'pred ssp585 r4i1p1f1'
label2 = 'pred ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    vmax=1,
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    vmax=1,
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP ctbias (~0)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "fut_ctbias_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()




# ctvar ----------------------------------------------------------------------------------------------------
# Now let's take a look at the NS
files = ["./exp0/data/error_metrics/CCCma_CanESM5_ssp585_r4i1p1f1_Lmon_npp_gn_v20190429_ctvar.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
bench_data = my_fxns.get_hist_data_antiantartica(files)
bench_data["exp_en"] = bench_data["experiment"].astype(str) + " " + bench_data["ensemble"].astype(str)
bench_data["metric"] = "ctvar"
bench_data["source"] = "benchmark"
bench_data["exp_en"] = "benchmark"

files = [ERROR_DIR+"npp_r4i1p1f1_ctvar1.nc",
         ERROR_DIR+"npp_r4i1p1f1_ctvar2.nc",
         ERROR_DIR+"npp_r10i1p1f1_ctvar1.nc",
         ERROR_DIR+"npp_r10i1p1f1_ctvar2.nc"]

# Read in the data, note that since this is the MSE and not the RMSE we will need to take the sqrt
futu_data = my_fxns.get_hist_data2(files)
futu_data["exp_en"] = futu_data["ensemble"].astype(str) + futu_data["metric"].astype(str)
futu_data["metric"] = "ctvar"
futu_data["source"] = "LPS"

to_plot = pd.concat([bench_data, futu_data], ignore_index=True)
to_plot['value'] = np.log10(to_plot['value'])
title_name = "log" + to_plot.metric[0] + " " + to_plot.variable[0] + " (~1)"
metric_name = to_plot.metric[0]

sns.histplot(
    data=to_plot,
    x="value",
    hue="exp_en",
    bins=500,
    multiple="layer",
)
plt.xlabel(metric_name)
plt.title(title_name)
plt.savefig(FIG_DIR + "fut_ctvar_npp.png", dpi=300, bbox_inches='tight')
plt.show()


to_plot = pd.concat([bench_data, futu_data], ignore_index=True)
summary_table = to_plot.groupby('exp_en')['value'].describe()
summary_table = summary_table[['mean', 'std', 'min', 'max']]
summary_table.to_csv(FIG_DIR + 'fut_ctvar_npp.csv', float_format='%.2e')

# Now let's just make some maps of the ctbais
d1 = xr.open_dataset(ERROR_DIR+"npp_r4i1p1f1_ctvar1.nc")
d2 = xr.open_dataset(ERROR_DIR+"npp_r10i1p1f1_ctvar2.nc")
d1 = d1.where(d1.lat > -60)
d2 = d2.where(d2.lat > -60)

# Now plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
VAR = 'npp'
label1 = 'pred ssp585 r4i1p1f1'
label2 = 'pred ssp585 r10i1p1f1'

d1[VAR].plot(
    ax=ax1,
    cmap="viridis",  # choose your colormap
    vmax=2,
    add_colorbar=True
)
ax1.set_title(label1, fontsize=12)


d2[VAR].plot(
    ax=ax2,
    cmap="viridis",  # choose your colormap
    vmax=2,
    add_colorbar=True
)
ax2.set_title(label2, fontsize=12)

fig.suptitle("NPP ctvar (~1)", fontsize=18, y=1.00)
plt.savefig(FIG_DIR + "fut_ctvar_npp_map.png", dpi=300, bbox_inches='tight')
plt.show()
